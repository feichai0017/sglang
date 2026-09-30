"""Unit tests for the deferred decode-side KV release mechanism.

When a decode request is aborted while its prefill->decode KV transfer may still
be in flight, the decode side holds its KV pages / req-slot instead of freeing
them immediately (which could let the still-in-flight write land on pages already
reused by another request). The pages are released once every prefill rank acks
that its transfer drained (CommonKVManager.is_abort_release_safe). Timeouts
retain the pages. See DecodeTransferQueue.resolve_deferred_releases.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation import decode as decode_mod
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_manager():
    """A bare CommonKVManager carrying only the deferred-ack state the helpers
    touch (avoids the heavy real __init__)."""
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr._deferred_abort_ack_tracker = {}
    return mgr


class TestAbortAckAggregation(CustomTestCase):
    def test_release_safe_only_after_all_required_ranks_ack(self):
        mgr = _make_manager()
        room = 100
        tokens = mgr.register_deferred_abort_room(room, 2)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))

        mgr.note_abort_ack(room, 0, tokens[0])
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))

        mgr.note_abort_ack(room, 1, tokens[1])
        self.assertTrue(mgr.is_abort_release_safe(room, required_acks=2))

    def test_duplicate_rank_ack_does_not_over_count(self):
        mgr = _make_manager()
        room = 101
        tokens = mgr.register_deferred_abort_room(room, 2)
        mgr.note_abort_ack(room, 0, tokens[0])
        mgr.note_abort_ack(room, 0, tokens[0])  # same rank twice
        # Two acks arrived but from one rank: not safe for a 2-rank prefill.
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))

    def test_single_rank_fast_path(self):
        mgr = _make_manager()
        room = 102
        tokens = mgr.register_deferred_abort_room(room, 1)
        mgr.note_abort_ack(room, 0, tokens[0])
        self.assertTrue(mgr.is_abort_release_safe(room, required_acks=1))

    def test_clear_deferred_abort_state(self):
        mgr = _make_manager()
        room = 103
        tokens = mgr.register_deferred_abort_room(room, 1)
        mgr.note_abort_ack(room, 0, tokens[0])
        mgr.clear_deferred_abort_state(room)
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=1))

    def test_ack_before_register_is_dropped(self):
        # An ack for a room that isn't actively held must not be recorded (it
        # would otherwise pollute a later request reusing the same room).
        mgr = _make_manager()
        room = 104
        mgr.note_abort_ack(room, 0, "old")  # no register yet
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=1))

    def test_late_ack_after_release_does_not_pollute_reused_room(self):
        # Regression for bootstrap_room reuse: req A (room R) releases, then a
        # late ack from A arrives, then req B reuses room R. B must start from a
        # clean slate and not inherit A's ack (which would release B early while
        # its transfer is still in flight -> KV corruption).
        mgr = _make_manager()
        room = 105

        # Req A: held, one of two ranks acks, then released (e.g. timed out).
        tokens = mgr.register_deferred_abort_room(room, 2)
        mgr.note_abort_ack(room, 0, tokens[0])
        mgr.clear_deferred_abort_state(room)

        # Late ack from A's other rank arrives after release -> dropped.
        mgr.note_abort_ack(room, 1, tokens[1])
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)

        # Req B reuses room R.
        tokens = mgr.register_deferred_abort_room(room, 2)
        # Only B's rank-0 has acked so far; a 2-rank prefill is NOT safe yet.
        mgr.note_abort_ack(room, 0, tokens[0])
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))
        mgr.note_abort_ack(room, 1, tokens[1])
        self.assertTrue(mgr.is_abort_release_safe(room, required_acks=2))

    def test_register_rejects_reuse_while_pages_are_held(self):
        mgr = _make_manager()
        mgr.register_deferred_abort_room(106, 2)
        with self.assertRaisesRegex(RuntimeError, "awaits drain"):
            mgr.register_deferred_abort_room(106, 2)

    def test_old_ack_cannot_complete_new_attempt_after_room_reuse(self):
        mgr = _make_manager()
        old = mgr.register_deferred_abort_room(106, 1)[0]
        mgr.note_abort_ack(106, 0, old)
        mgr.clear_deferred_abort_state(106)
        new = mgr.register_deferred_abort_room(106, 1)[0]
        mgr.note_abort_ack(106, 0, old)
        self.assertFalse(mgr.is_abort_release_safe(106, 1))
        mgr.note_abort_ack(106, 0, new)
        self.assertTrue(mgr.is_abort_release_safe(106, 1))


class _FakeIdxAllocator:
    def __init__(self):
        self.freed = []

    def free(self, idx):
        self.freed.append(idx)


def _make_queue(timeout=30.0):
    q = DecodeTransferQueue.__new__(DecodeTransferQueue)
    q.queue = []
    q._deferred_releases = []
    q.deferred_kv_release_timeout = timeout
    q.enable_staging = False
    q.staging_handler = None
    q.tree_cache = object()
    q.metadata_buffers = SimpleNamespace(bootstrap_room={})
    q.req_to_metadata_buffer_idx_allocator = _FakeIdxAllocator()
    return q


def _make_decode_req(room, idx, mgr, n_prefill_ranks=1):
    receiver = SimpleNamespace(
        kv_mgr=mgr,
        abort_requested=True,
        # One entry per prefill rank the decode notified of the abort; its length
        # is the required drain-ack count (see DecodeTransferQueue._defer_release).
        bootstrap_infos=[{"rank": r} for r in range(n_prefill_ranks)],
        clear=lambda: None,
        _send_abort_notification=lambda: None,
    )
    return SimpleNamespace(
        req=SimpleNamespace(bootstrap_room=room),
        kv_receiver=receiver,
        metadata_buffer_index=idx,
        transfer_observation=None,
        transfer_failure="cancelled",
    )


class TestResolveDeferredReleases(CustomTestCase):
    def test_noop_when_nothing_deferred(self):
        q = _make_queue()
        with patch.object(decode_mod, "release_kv_cache") as rel:
            q.resolve_deferred_releases()
        rel.assert_not_called()

    def test_holds_until_drained_then_releases(self):
        mgr = _make_manager()
        room, idx = 200, 7
        q = _make_queue()
        dreq = _make_decode_req(room, idx, mgr, n_prefill_ranks=2)
        # In production the room is armed in abort_request when the ABORT is
        # sent, before the scheduler defers here.
        tokens = mgr.register_deferred_abort_room(room, 2)
        q._defer_release(dreq)

        with patch.object(decode_mod, "release_kv_cache") as rel:
            # Not yet acked -> held, not released.
            q.resolve_deferred_releases()
            rel.assert_not_called()
            self.assertEqual(len(q._deferred_releases), 1)

            # One of two ranks acked -> still held.
            mgr.note_abort_ack(room, 0, tokens[0])
            q.resolve_deferred_releases()
            rel.assert_not_called()
            self.assertEqual(len(q._deferred_releases), 1)

            # Both ranks acked -> released exactly once.
            mgr.note_abort_ack(room, 1, tokens[1])
            q.resolve_deferred_releases()
            rel.assert_called_once_with(dreq.req, q.tree_cache, is_insert=False)

        # Held state fully cleaned up.
        self.assertEqual(q._deferred_releases, [])
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [idx])
        self.assertEqual(q.metadata_buffers.bootstrap_room[idx], 0)
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)
        self.assertIsNone(dreq.kv_receiver)

    def test_timeout_keeps_pages_and_blocks_memory_unload_until_late_ack(self):
        mgr = _make_manager()
        room, idx = 300, 3
        q = _make_queue(timeout=30.0)
        dreq = _make_decode_req(room, idx, mgr)
        token = mgr.register_deferred_abort_room(room, 1)[0]
        q._deferred_releases.append((dreq, float("-inf"), idx, 1))
        with patch.object(decode_mod, "release_kv_cache") as rel:
            q.resolve_deferred_releases()
            rel.assert_not_called()
            self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [])
            with self.assertRaisesRegex(RuntimeError, "undrained"):
                q.release_memory_occupation()
            mgr.note_abort_ack(room, 0, token)
            q.resolve_deferred_releases()
            rel.assert_called_once_with(dreq.req, q.tree_cache, is_insert=False)
            q.release_memory_occupation()
        self.assertEqual(q._deferred_releases, [])
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [idx])
        self.assertIsNone(dreq.kv_receiver)

    def test_failed_release_is_isolated_and_not_retried(self):
        # A raising _do_release must drop the entry (no double-free on retry) and
        # not brick resolve for the remaining entries or subsequent calls.
        mgr = _make_manager()
        q = _make_queue()
        good = _make_decode_req(700, 1, mgr)
        bad = _make_decode_req(701, 2, mgr)
        for room in (700, 701):
            token = mgr.register_deferred_abort_room(room, 1)[0]
            mgr.note_abort_ack(room, 0, token)
        # Both have proved drain, so release failures cannot affect safety.
        q._deferred_releases.append((bad, float("-inf"), 2, 1))
        q._deferred_releases.append((good, float("-inf"), 1, 1))

        calls = []

        def fake_release(req, tree_cache, is_insert):
            calls.append(req)
            if req is bad.req:
                raise RuntimeError("boom")

        with patch.object(decode_mod, "release_kv_cache", side_effect=fake_release):
            q.resolve_deferred_releases()  # must not raise
            # The good one still released despite the bad one throwing.
            self.assertIn(good.req, calls)
            # Nothing left held, and a second call is a clean no-op (no retry).
            self.assertEqual(q._deferred_releases, [])
            q.resolve_deferred_releases()

    def test_defer_release_records_deadline_and_idx(self):
        mgr = _make_manager()
        q = _make_queue(timeout=12.5)
        dreq = _make_decode_req(room=400, idx=9, mgr=mgr)
        q._defer_release(dreq)
        self.assertEqual(len(q._deferred_releases), 1)
        held_req, deadline, held_idx, required = q._deferred_releases[0]
        self.assertIs(held_req, dreq)
        self.assertEqual(held_idx, 9)
        self.assertIsInstance(deadline, float)


if __name__ == "__main__":
    unittest.main()
