"""Decode-owned observations and partial-submit drain are public contracts."""

import concurrent.futures
import threading
import unittest
from unittest.mock import Mock, patch

from sglang.srt.disaggregation import lifecycle
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPublicLifecycle(unittest.TestCase):
    def test_observer_failure_cannot_change_request_lifecycle(self):
        with patch.dict(lifecycle._observers, {}, clear=True):
            failing = Mock(side_effect=RuntimeError("observer failed"))
            events = []
            lifecycle.register_pd_transfer_observer("failing", failing)
            lifecycle.register_pd_transfer_observer("recording", events.append)
            event = lifecycle.observe_admission(
                request_id="r",
                bootstrap_room=9,
                source_endpoint="p:30000",
                logical_bytes=4096,
                fragment_count=2,
                queue_depth=1,
            )
            lifecycle.observe_transition(event, "quarantined")
            lifecycle.observe_transition(event, "cancelled")
            self.assertEqual(
                [e.phase for e in events], ["admitted", "quarantined", "cancelled"]
            )
            self.assertEqual(len({e.transfer_id for e in events}), 1)
            self.assertGreater(events[-1].elapsed_ns, 0)
            failing.assert_called_once()

    def test_abort_wire_ack_echoes_this_attempt_token(self):
        mgr = CommonKVManager.__new__(CommonKVManager)
        mgr._prefill_unique_rank = lambda: 2
        mgr._send_multipart_locked = Mock()
        mgr._send_abort_ack("127.0.0.1", 30100, 9, "attempt-generation")
        self.assertEqual(
            mgr._send_multipart_locked.call_args.args[1],
            [b"ABORT_ACK", b"9", b"2", b"attempt-generation"],
        )

    def test_partial_layer_submission_drains_the_first_running_write(self):
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        started, release, finished = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

            def write_layer(index):
                started.set()
                assert release.wait(5)
                finished.set()
                return 0

            def submit(worker, index):
                if index == 1:
                    assert started.wait(5)
                    raise RuntimeError("second submit rejected")
                return executor.submit(worker, index)

            submitter = Mock(submit=submit)
            operation = executor.submit(
                mgr._submit_transfer_layers, submitter, write_layer, [(0,), (1,)]
            )
            self.assertTrue(started.wait(5))
            self.assertFalse(operation.done())
            self.assertFalse(finished.is_set())
            release.set()
            self.assertEqual(operation.result(timeout=5), -1)
            self.assertTrue(finished.is_set())


if __name__ == "__main__":
    unittest.main()
