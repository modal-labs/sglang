"""CPU protocol-model tests for the CUDA IPC lease pool."""

import random
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from sglang.srt.multimodal.transport import lease_guard
from sglang.srt.multimodal.transport.lease_guard import StaleLeaseError
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@dataclass
class MockSlot:
    generation: int = 0
    ready_word: int = 0
    ack_words: list[int] | None = None
    byte_range: tuple[int, int] = (0, 0)
    state: str = "FREE"


class MockProxy:
    def __init__(self, pool, slot: int, generation: int, nbytes: int):
        self.proxy_state = {"ipc_extra": {"pool_handle": pool.handle}}
        self.ready_byte_offset = slot * pool.slot_stride
        self.total_consumer_count = pool.ranks
        self.generation = generation
        self.slot = slot
        self.nbytes = nbytes


class MockLeasePool:
    _pool_counter = 0

    def __init__(self, ranks: int, slots: int):
        self.ranks = ranks
        type(self)._pool_counter += 1
        self.handle = ("mock", type(self)._pool_counter, ranks, slots)
        self.slot_stride = (1 + ranks) * 4
        self.slots = [
            MockSlot(ack_words=[0] * ranks, byte_range=(i * 1024, (i + 1) * 1024))
            for i in range(slots)
        ]
        self._next_generation = 0
        self.recycle_count = 0

    def lease(self, nbytes: int):
        for slot_id, slot in enumerate(self.slots):
            if slot.state != "FREE":
                continue
            self._next_generation += 1
            slot.generation = self._next_generation
            slot.ready_word = 0
            slot.byte_range = (slot_id * 1024, slot_id * 1024 + nbytes)
            slot.state = "ACTIVE"
            return MockProxy(self, slot_id, slot.generation, nbytes)
        return None

    def clone(self, proxy: MockProxy):
        return MockProxy(self, proxy.slot, proxy.generation, proxy.nbytes)

    def publish(self, proxy: MockProxy):
        slot = self.slots[proxy.slot]
        assert slot.state == "ACTIVE"
        assert slot.generation == proxy.generation
        slot.ready_word = proxy.generation

    def read(self, proxy: MockProxy):
        slot = self.slots[proxy.slot]
        if slot.ready_word < proxy.generation or slot.state != "ACTIVE":
            raise RuntimeError("lease is not published or active")
        proxy.proxy_state["ipc_extra"]["pool_handle"] = (self.handle, 0)
        lease_guard.check_read(proxy, rank=0)
        assert slot.generation == proxy.generation, (
            f"guard allowed stale read: slot generation {slot.generation}, "
            f"proxy generation {proxy.generation}"
        )
        assert slot.state == "ACTIVE"
        return proxy.nbytes

    def ack(self, proxy: MockProxy, rank: int, *, guarded: bool = True):
        slot = self.slots[proxy.slot]
        if slot.ready_word < proxy.generation:
            raise RuntimeError("lease is not published")
        proxy.proxy_state["ipc_extra"]["pool_handle"] = (self.handle, rank)
        if guarded and not lease_guard.check_and_record_write(proxy, rank=rank):
            return False
        slot.ack_words[rank] = proxy.generation
        return True

    def ack_all(self, proxy: MockProxy):
        slot = self.slots[proxy.slot]
        for rank in range(self.ranks):
            slot.ack_words[rank] = proxy.generation

    def cancel(self, proxy: MockProxy):
        slot = self.slots[proxy.slot]
        if slot.state != "ACTIVE" or slot.generation != proxy.generation:
            return
        slot.ack_words = [proxy.generation] * self.ranks
        slot.state = "CANCELLED"

    def recycle(self, slot_id: int):
        slot = self.slots[slot_id]
        if slot.state == "ACTIVE" and all(
            value == slot.generation for value in slot.ack_words
        ):
            slot.state = "FREE"
            self.recycle_count += 1

    def assert_no_overlaps(self):
        active = [slot.byte_range for slot in self.slots if slot.state == "ACTIVE"]
        for index, (start, end) in enumerate(active):
            for other_start, other_end in active[index + 1 :]:
                assert end <= other_start or other_end <= start


def _drain(pool, proxies):
    for slot_id, slot in enumerate(pool.slots):
        if slot.state != "ACTIVE":
            continue
        current = next(
            proxy
            for proxy in reversed(proxies)
            if proxy.slot == slot_id and proxy.generation == slot.generation
        )
        for rank in range(pool.ranks):
            pool.ack(current, rank)
        pool.recycle(slot_id)


def test_randomized_own_word_protocol_invariants():
    lease_guard.reset()
    for seed in range(200):
        rng = random.Random(seed)
        ranks = rng.choice((1, 2, 8))
        pool = MockLeasePool(ranks, rng.randint(1, 3))
        proxies = []
        previous_acks = {}

        for _ in range(300):
            pool.assert_no_overlaps()
            for slot_id, slot in enumerate(pool.slots):
                for rank, ack in enumerate(slot.ack_words):
                    key = (slot_id, rank)
                    assert ack >= previous_acks.get(key, 0)
                    previous_acks[key] = ack

            active = [
                proxy
                for proxy in proxies
                if pool.slots[proxy.slot].state == "ACTIVE"
                and pool.slots[proxy.slot].generation == proxy.generation
            ]
            stale = [proxy for proxy in proxies if proxy not in active]

            if stale and rng.random() < 0.2:
                proxy = rng.choice(stale)
                if rng.random() < 0.5:
                    try:
                        pool.read(proxy)
                    except (StaleLeaseError, RuntimeError):
                        pass
                else:
                    pool.ack(proxy, rng.randrange(ranks))
            elif active and rng.random() < 0.8:
                proxy = rng.choice(active)
                if rng.random() < 0.35:
                    try:
                        pool.read(proxy)
                    except (StaleLeaseError, RuntimeError):
                        pass
                else:
                    pool.ack(proxy, rng.randrange(ranks))
            else:
                proxy = pool.lease(rng.randint(1, 64))
                if proxy is not None:
                    proxies.append(proxy)
                    pool.publish(proxy)
                    if rng.random() < 0.5:
                        proxies.append(pool.clone(proxy))
                    if rng.random() < 0.15:
                        pool.cancel(proxy)

            for slot_id in range(len(pool.slots)):
                pool.recycle(slot_id)

        _drain(pool, proxies)
        assert all(slot.state in {"FREE", "CANCELLED"} for slot in pool.slots), (
            seed,
            [(slot.state, slot.generation, slot.ack_words) for slot in pool.slots],
        )
        pool.assert_no_overlaps()

    assert lease_guard.stats["stale_read_refused"] > 0
    assert lease_guard.stats["stale_write_refused"] > 0


def test_strict_upstream_trace_leaks_but_own_word_guard_recycles():
    def run(strict: bool):
        lease_guard.reset()
        pool = MockLeasePool(ranks=2, slots=1)
        first = pool.lease(8)
        pool.publish(first)
        if strict:
            pool.read(first)
            pool.ack_all(first)
        else:
            pool.ack(first, 0)
        pool.ack(first, 1)
        pool.recycle(0)

        second = pool.lease(8)
        pool.publish(second)
        pool.ack(second, 0)
        pool.ack(second, 1)
        if strict:
            pool.ack(first, 1, guarded=False)
        else:
            assert not pool.ack(first, 1)
        pool.recycle(0)
        return pool

    strict_pool = run(strict=True)
    assert strict_pool.slots[0].state == "ACTIVE"
    assert strict_pool.slots[0].ack_words == [2, 1]

    guarded_pool = run(strict=False)
    assert guarded_pool.slots[0].state == "FREE"
    assert lease_guard.stats["stale_write_refused"] == 1


def test_negative_control_guard_disabled_detects_stale_read():
    rng = random.Random(17)
    pool = MockLeasePool(ranks=2, slots=1)
    first = pool.lease(rng.randint(1, 64))
    pool.publish(first)
    for rank in range(pool.ranks):
        pool.ack(first, rank)
    pool.recycle(0)

    second = pool.lease(rng.randint(1, 64))
    pool.publish(second)
    with (
        patch.object(lease_guard, "check_read"),
        pytest.raises(AssertionError, match="guard allowed stale read"),
    ):
        pool.read(first)
