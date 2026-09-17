//! Opt-in single-pair HTTP PD admission. Text overlap is an estimate, not engine KV state.
//! Every waiter (including the idle fast path) participates in the same ordering.
//! A lease is held through the full response body, not merely response headers.
use crate::policies::tree::Tree;
use axum::{body::Body, response::Response};
use futures_util::Stream;
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::{oneshot, Notify};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Order {
    Fifo,
    Hrrn,
}
struct Entry {
    id: u64,
    arrived: Instant,
    key: Arc<str>,
    bytes: usize,
    sender: oneshot::Sender<Lease>,
}
struct State {
    next: u64,
    active: usize,
    bytes: usize,
    entries: Vec<Entry>,
    offered: u64,
    dispatched: u64,
    rejected: u64,
    cancelled: u64,
    wait_sum: f64,
    wait_buckets: [u64; 16],
    completed: Vec<Arc<str>>,
}
pub struct Admission {
    state: Mutex<State>,
    notify: Arc<Notify>,
    started: AtomicBool,
    dispatch: Mutex<DispatchState>,
    tree: Tree,
    order: Order,
    slots: usize,
    max_waiters: usize,
    max_bytes: usize,
    timeout: Duration,
}
#[derive(Default)]
struct DispatchState {
    version: u64,
    costs: HashMap<u64, (u64, usize)>,
}
impl Drop for Admission {
    fn drop(&mut self) {
        self.notify.notify_one();
    }
}
impl std::fmt::Debug for Admission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Admission")
            .field("order", &self.order)
            .field("slots", &self.slots)
            .finish()
    }
}
const WAIT_BUCKETS: [f64; 16] = [
    0.001,
    0.01,
    0.05,
    0.1,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    f64::INFINITY,
];
impl Admission {
    pub fn render_metrics(&self) -> String {
        use std::fmt::Write;
        let s = self.state.lock();
        let mut out = String::new();
        for (name, value) in [
            ("active", s.active),
            ("waiting", s.entries.len()),
            ("queued_bytes", s.bytes),
        ] {
            writeln!(out, "# TYPE smg:pd_admission_{name} gauge").unwrap();
            writeln!(out, "smg:pd_admission_{name} {value}").unwrap();
        }
        for (name, value) in [
            ("offered", s.offered),
            ("dispatched", s.dispatched),
            ("rejected", s.rejected),
            ("cancelled_or_expired", s.cancelled),
        ] {
            writeln!(out, "# TYPE smg:pd_admission_{name}_total counter").unwrap();
            writeln!(out, "smg:pd_admission_{name}_total {value}").unwrap();
        }
        writeln!(out, "# TYPE smg:pd_admission_wait_seconds histogram").unwrap();
        for (bound, count) in WAIT_BUCKETS.iter().zip(s.wait_buckets.iter()) {
            let le = if bound.is_infinite() {
                "+Inf".into()
            } else {
                bound.to_string()
            };
            writeln!(
                out,
                "smg:pd_admission_wait_seconds_bucket{{le=\"{le}\"}} {count}"
            )
            .unwrap();
        }
        writeln!(out, "smg:pd_admission_wait_seconds_sum {}", s.wait_sum).unwrap();
        writeln!(out, "smg:pd_admission_wait_seconds_count {}", s.dispatched).unwrap();
        out
    }
    pub fn from_env() -> Result<Option<Arc<Self>>, String> {
        let Ok(policy) = std::env::var("SMG_PD_ADMISSION_POLICY") else {
            return Ok(None);
        };
        let order = match policy.as_str() {
            "fifo" => Order::Fifo,
            "hrrn" => Order::Hrrn,
            _ => return Err("SMG_PD_ADMISSION_POLICY must be fifo or hrrn".into()),
        };
        fn number(name: &str, default: usize) -> Result<usize, String> {
            let n = std::env::var(name).map_or(Ok(default), |v| {
                v.parse::<usize>().map_err(|_| format!("Invalid {name}"))
            })?;
            if n == 0 {
                return Err(format!("{name} must be positive"));
            }
            Ok(n)
        }
        let slots = number("SMG_PD_ADMISSION_MAX_INFLIGHT", 64)?;
        let waiters = number("SMG_PD_ADMISSION_MAX_WAITERS", 256)?;
        let bytes = number("SMG_PD_ADMISSION_MAX_BYTES", 268435456)?;
        let timeout = number("SMG_PD_ADMISSION_TIMEOUT_SECONDS", 600)?;
        tracing::info!(
            policy,
            slots,
            waiters,
            bytes,
            timeout,
            "PD admission enabled (single pair, approximate text cost)"
        );
        Ok(Some(Self::new(
            order,
            slots,
            waiters,
            bytes,
            Duration::from_secs(timeout as u64),
        )))
    }
    pub(super) fn new(
        order: Order,
        slots: usize,
        max_waiters: usize,
        max_bytes: usize,
        timeout: Duration,
    ) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State {
                next: 0,
                active: 0,
                bytes: 0,
                entries: Vec::new(),
                offered: 0,
                dispatched: 0,
                rejected: 0,
                cancelled: 0,
                wait_sum: 0.0,
                wait_buckets: [0; 16],
                completed: Vec::new(),
            }),
            notify: Arc::new(Notify::new()),
            started: AtomicBool::new(false),
            dispatch: Mutex::new(DispatchState::default()),
            tree: Tree::new(),
            order,
            slots,
            max_waiters,
            max_bytes,
            timeout,
        })
    }
    fn start_dispatcher(self: &Arc<Self>) {
        if self.started.swap(true, Ordering::AcqRel) {
            return;
        }
        let weak = Arc::downgrade(self);
        let notify = self.notify.clone();
        tokio::spawn(async move {
            loop {
                notify.notified().await;
                let Some(owner) = weak.upgrade() else { break };
                // Prefix matching and cache insertion must not block Tokio workers.
                if let Err(error) = tokio::task::spawn_blocking(move || owner.dispatch_once()).await
                {
                    tracing::error!(?error, "PD admission dispatcher failed");
                    break;
                }
            }
        });
    }

    fn dispatch_once(self: &Arc<Self>) {
        // Exactly one dispatcher. HTTP handlers/metrics never acquire this lock.
        let mut dispatch = self.dispatch.lock();
        let (completed, entries) = {
            let mut state = self.state.lock();
            let completed = std::mem::take(&mut state.completed);
            let entries = state
                .entries
                .iter()
                .map(|e| (e.id, e.arrived, e.key.clone()))
                .collect::<Vec<_>>();
            (completed, entries)
        };
        if !completed.is_empty() && self.order == Order::Hrrn {
            for key in completed {
                self.tree.insert(&key, "completed");
            }
            if self
                .tree
                .get_tenant_char_count()
                .get("completed")
                .copied()
                .unwrap_or(0)
                > 16 * 1024 * 1024
            {
                self.tree.evict_tenant_by_size(16 * 1024 * 1024);
            }
            dispatch.version += 1;
        }
        dispatch
            .costs
            .retain(|id, _| entries.iter().any(|e| e.0 == *id));
        if self.state.lock().active >= self.slots {
            return;
        }
        let version = dispatch.version;
        let now = Instant::now();
        // Match each text once per cache version, outside the request-state lock.
        // Sorting uses only numeric scores, never prefix matching in a comparator.
        let mut ranked = entries
            .iter()
            .map(|(id, arrived, key)| {
                let cost = if self.order == Order::Hrrn {
                    if dispatch.costs.get(id).is_none_or(|(v, _)| *v != version) {
                        let m = self.tree.prefix_match_with_counts(key);
                        dispatch.costs.insert(
                            *id,
                            (
                                version,
                                m.input_char_count
                                    .saturating_sub(m.matched_char_count)
                                    .max(1),
                            ),
                        );
                    }
                    dispatch.costs[id].1
                } else {
                    1
                };
                (
                    *id,
                    now.duration_since(*arrived).as_secs_f64() / cost as f64,
                )
            })
            .collect::<Vec<_>>();
        ranked.sort_unstable_by(|a, b| {
            if self.order == Order::Fifo {
                a.0.cmp(&b.0)
            } else {
                b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
            }
        });
        let mut selected = Vec::new();
        {
            let mut state = self.state.lock();
            for (id, _) in ranked {
                if state.active >= self.slots {
                    break;
                }
                let Some(pos) = state.entries.iter().position(|e| e.id == id) else {
                    continue;
                };
                // The waiter owns deadline/cancellation cleanup.
                if state.entries[pos].sender.is_closed()
                    || state.entries[pos].arrived.elapsed() >= self.timeout
                {
                    continue;
                }
                let entry = state.entries.remove(pos);
                state.bytes -= entry.bytes;
                state.active += 1;
                state.dispatched += 1;
                let wait = entry.arrived.elapsed().as_secs_f64();
                state.wait_sum += wait;
                for (i, bound) in WAIT_BUCKETS.iter().enumerate() {
                    if wait <= *bound {
                        state.wait_buckets[i] += 1;
                    }
                }
                selected.push((entry, wait));
            }
        }
        for (entry, wait) in selected {
            let lease = Lease {
                owner: self.clone(),
                key: entry.key,
                cache_success: false,
                wait_seconds: wait,
            };
            // Sending/dropping a lease can release a slot; never do it under state.
            let _ = entry.sender.send(lease);
            metrics::histogram!("smg_pd_admission_wait_seconds").record(wait);
        }
    }

    pub async fn acquire(
        self: &Arc<Self>,
        key: String,
        bytes: usize,
    ) -> Result<Lease, &'static str> {
        let arrived = Instant::now();
        let (sender, receiver) = oneshot::channel();
        let id = {
            let mut s = self.state.lock();
            s.offered += 1;
            if s.entries.len() >= self.max_waiters || bytes > self.max_bytes.saturating_sub(s.bytes)
            {
                s.rejected += 1;
                return Err("PD admission queue is full");
            }
            let id = s.next;
            s.next += 1;
            s.bytes += bytes;
            s.entries.push(Entry {
                id,
                arrived,
                key: Arc::from(key),
                bytes,
                sender,
            });
            id
        };
        let mut ticket = Ticket {
            owner: self.clone(),
            id,
            queued: true,
        };
        self.start_dispatcher();
        self.notify.notify_one();
        match tokio::time::timeout(self.timeout, receiver).await {
            Ok(Ok(lease)) => {
                ticket.queued = false;
                tracing::info!(
                    queue_wait_s = arrived.elapsed().as_secs_f64(),
                    "PD admission dispatch"
                );
                Ok(lease)
            }
            Ok(Err(_)) => Err("PD admission dispatcher stopped"),
            Err(_) => Err("PD admission queue deadline exceeded"),
        }
    }
}
struct Ticket {
    owner: Arc<Admission>,
    id: u64,
    queued: bool,
}
impl Drop for Ticket {
    fn drop(&mut self) {
        if self.queued {
            let mut s = self.owner.state.lock();
            if let Some(i) = s.entries.iter().position(|e| e.id == self.id) {
                let e = s.entries.remove(i);
                s.bytes -= e.bytes;
                s.cancelled += 1;
            }
            drop(s);
            self.owner.notify.notify_one();
        }
    }
}
pub struct Lease {
    pub(super) wait_seconds: f64,
    owner: Arc<Admission>,
    key: Arc<str>,
    cache_success: bool,
}
impl Drop for Lease {
    fn drop(&mut self) {
        {
            let mut state = self.owner.state.lock();
            if self.cache_success && self.owner.order == Order::Hrrn {
                state.completed.push(self.key.clone());
            }
            state.active -= 1;
        }
        self.owner.notify.notify_one();
    }
}
struct LeasedStream<S> {
    inner: S,
    lease: Option<Lease>,
    success: bool,
}
impl<S: Stream<Item = Result<bytes::Bytes, axum::Error>> + Unpin> Stream for LeasedStream<S> {
    type Item = Result<bytes::Bytes, axum::Error>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let result = std::pin::Pin::new(&mut self.inner).poll_next(cx);
        if let std::task::Poll::Ready(None) = &result {
            if let Some(mut lease) = self.lease.take() {
                lease.cache_success = self.success;
            }
        } else if matches!(&result, std::task::Poll::Ready(Some(Err(_)))) {
            self.lease.take();
        }
        result
    }
}
pub fn hold_response(response: Response, lease: Lease) -> Response {
    let success = response.status().is_success();
    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::from_stream(LeasedStream {
            inner: body.into_data_stream(),
            lease: Some(lease),
            success,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn lease_carries_its_own_queue_wait() {
        let q = Admission::new(Order::Fifo, 1, 4, 1000, Duration::from_secs(5));
        let first = q.acquire("first".into(), 5).await.unwrap();
        let q2 = q.clone();
        let queued = tokio::spawn(async move { q2.acquire("second".into(), 6).await.unwrap() });
        tokio::time::sleep(Duration::from_millis(50)).await;
        drop(first);
        let second = queued.await.unwrap();
        assert!(second.wait_seconds >= 0.04);
        assert!(second.wait_seconds < 5.0);
    }

    #[tokio::test]
    async fn hard_bound_cancellation_and_stream_lifetime() {
        let q = Admission::new(Order::Fifo, 1, 1, 100, Duration::from_secs(2));
        let first = q.acquire("a".into(), 1).await.unwrap();
        let q2 = q.clone();
        let waiter = tokio::spawn(async move { q2.acquire("b".into(), 1).await });
        tokio::task::yield_now().await;
        assert!(q.acquire("c".into(), 1).await.is_err());
        waiter.abort();
        let _ = waiter.await;
        assert_eq!(q.state.lock().entries.len(), 0);
        let body = Body::from_stream(futures_util::stream::pending::<
            Result<bytes::Bytes, axum::Error>,
        >());
        let response = hold_response(Response::new(body), first);
        assert_eq!(q.state.lock().active, 1);
        drop(response);
        assert_eq!(q.state.lock().active, 0);
        assert!(q.acquire("d".into(), 1).await.is_ok());
    }
    #[tokio::test]
    async fn timeout_and_bytes_released() {
        let q = Admission::new(Order::Fifo, 1, 2, 10, Duration::from_millis(10));
        let _first = q.acquire("a".into(), 1).await.unwrap();
        assert!(q.acquire("b".into(), 11).await.is_err());
        assert!(q.acquire("b".into(), 10).await.is_err());
        assert_eq!(q.state.lock().bytes, 0);
    }
    #[tokio::test]
    async fn hrrn_refreshes_overlap_and_fifo_preserves_arrival() {
        for order in [Order::Hrrn, Order::Fifo] {
            let q = Admission::new(order, 1, 4, 1000, Duration::from_secs(5));
            let mut first = q.acquire("abcdefghi".into(), 9).await.unwrap();
            let q2 = q.clone();
            let a = tokio::spawn(async move { q2.acquire("abcdefghij".into(), 10).await });
            tokio::time::sleep(Duration::from_millis(10)).await;
            let q2 = q.clone();
            let b = tokio::spawn(async move { q2.acquire("zzzzzz".into(), 6).await });
            tokio::time::sleep(Duration::from_millis(10)).await;
            first.cache_success = true;
            drop(first);
            let lease = tokio::time::timeout(Duration::from_secs(1), a)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert!(!b.is_finished());
            drop(lease);
            drop(b.await.unwrap().unwrap());
        }
    }
    #[tokio::test]
    async fn long_prompt_queue_fills_free_slots_without_blocking_metrics() {
        let q = Admission::new(
            Order::Hrrn,
            64,
            256,
            256 * 1024 * 1024,
            Duration::from_secs(30),
        );
        let mut held = Vec::new();
        for _ in 0..64 {
            held.push(q.acquire("busy".into(), 4).await.unwrap());
        }
        let prefix = "a".repeat(256 * 1024);
        q.tree.insert(&prefix, "completed");
        let mut waiters = Vec::new();
        for i in 0..256 {
            let owner = q.clone();
            let key = format!("{prefix}{i}");
            waiters.push(tokio::spawn(
                async move { owner.acquire(key, 512 * 1024).await },
            ));
        }
        tokio::time::timeout(Duration::from_secs(5), async {
            while q.state.lock().entries.len() != 256 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let started = Instant::now();
        drop(held);
        let mut worst_metrics = Duration::ZERO;
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let t = Instant::now();
                let _ = q.render_metrics();
                worst_metrics = worst_metrics.max(t.elapsed());
                if waiters.iter().filter(|w| w.is_finished()).count() >= 64 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(q.state.lock().active, 64);
        assert_eq!(q.state.lock().entries.len(), 192);
        println!(
            "256 x 256KiB keys: fill64={:?}, worst_metrics={:?}",
            started.elapsed(),
            worst_metrics
        );
        for waiter in waiters {
            waiter.abort();
            drop(waiter.await);
        }
        tokio::time::timeout(Duration::from_secs(5), async {
            while q.state.lock().active != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(q.state.lock().bytes, 0);
    }

    #[tokio::test]
    async fn hrrn_chooses_short_work_and_does_not_rematch_unchanged_cache() {
        let q = Admission::new(Order::Hrrn, 1, 4, 1000, Duration::from_secs(5));
        let first = q.acquire("busy".into(), 4).await.unwrap();
        let owner = q.clone();
        let long = tokio::spawn(async move { owner.acquire("x".repeat(100), 100).await });
        tokio::time::sleep(Duration::from_millis(10)).await;
        let owner = q.clone();
        let short = tokio::spawn(async move { owner.acquire("z".into(), 1).await });
        tokio::time::sleep(Duration::from_millis(10)).await;
        drop(first);
        let selected = tokio::time::timeout(Duration::from_secs(1), short)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(!long.is_finished());
        let costs = q.dispatch.lock().costs.clone();
        q.notify.notify_one();
        tokio::time::sleep(Duration::from_millis(10)).await;
        for (id, value) in &q.dispatch.lock().costs {
            assert_eq!(costs.get(id), Some(value));
        }
        drop(selected);
        drop(long.await.unwrap().unwrap());
    }

    #[tokio::test]
    async fn completion_seeds_cache_error_and_disconnect_do_not() {
        let q = Admission::new(Order::Hrrn, 1, 4, 1000, Duration::from_secs(1));
        let r = hold_response(
            Response::new(Body::from("ok")),
            q.acquire("prefix".into(), 6).await.unwrap(),
        );
        let _ = axum::body::to_bytes(r.into_body(), 100).await.unwrap();
        assert_eq!(q.state.lock().active, 0);
        tokio::time::timeout(Duration::from_secs(1), async {
            while q
                .tree
                .prefix_match_with_counts("prefix!")
                .matched_char_count
                != 6
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(
            q.tree
                .prefix_match_with_counts("prefix!")
                .matched_char_count,
            6
        );
        let r = hold_response(
            Response::builder()
                .status(400)
                .body(Body::from("bad"))
                .unwrap(),
            q.acquire("other".into(), 5).await.unwrap(),
        );
        let _ = axum::body::to_bytes(r.into_body(), 100).await.unwrap();
        assert_eq!(
            q.tree.prefix_match_with_counts("other").matched_char_count,
            0
        );
    }
}
