"""Bounded FIFO backfill for decode reservation (canary opt-in).

A blocked head can be bypassed by at most its own reservation demand in tokens.
Credit persists across scheduling passes; it is not replenished by completions.
This bounds added allocation work ahead of that head, not wall-clock waiting.
"""


class DecodeBackfillBudget:
    def __init__(self):
        self.head = None
        self.total_admitted = 0
        self.total_tokens = 0
        self.demand = 0
        self.remaining = 0

    def begin(self, first_ready_rid):
        if self.head != first_ready_rid:
            self.head = None
            self.demand = self.remaining = 0

    def blocked(self, rid, demand):
        if self.head is None:
            self.head = rid
            self.demand = self.remaining = demand
        return self.remaining > 0

    def allows(self, rid, demand):
        return (
            self.head is None
            or rid == self.head
            or (demand < self.demand and demand <= self.remaining)
        )

    def admitted(self, rid, demand):
        if self.head is None:
            return
        if rid == self.head:
            self.head = None
            self.demand = self.remaining = 0
        else:
            assert 0 < demand <= self.remaining
            self.remaining -= demand
            self.total_admitted += 1
            self.total_tokens += demand
