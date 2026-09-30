"""Plugin construction must preserve a single, lazily initialized payload owner."""

import os
import unittest
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

from sglang.srt.distributed.device_communicators import (
    mooncake_transfer_engine as engines,
)


class TestMooncakeEngineFactory(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(engines, "_mooncake_transfer_engine", None))
        self.enterContext(patch.object(engines, "_mooncake_transfer_engine_name", None))
        self.enterContext(
            patch.dict(engines._mooncake_transfer_engine_factories, {}, clear=True)
        )
        self.enterContext(
            patch.dict(os.environ, {"SGLANG_MOONCAKE_TRANSFER_ENGINE": "test"})
        )

    def test_registered_factory_is_lazy_and_owns_the_shared_instance(self):
        factory = Mock()
        native_class = engines.MooncakeTransferEngine
        engines.register_mooncake_transfer_engine_factory("test", factory)
        engines.register_mooncake_transfer_engine_factory("test", factory)
        factory.assert_not_called()
        self.assertIsNone(engines.get_mooncake_transfer_engine())
        self.assertIs(engines.MooncakeTransferEngine, native_class)

        first = engines.init_mooncake_transfer_engine("host", 2, "mlx5_1")
        self.assertIs(first, factory.return_value)
        self.assertIs(engines.get_mooncake_transfer_engine(), first)
        self.assertIs(engines.init_mooncake_transfer_engine("host", 2, "mlx5_1"), first)
        factory.assert_called_once_with(hostname="host", gpu_id=2, ib_device="mlx5_1")

    def test_conflicting_plugin_cannot_replace_an_owner(self):
        first = Mock()
        engines.register_mooncake_transfer_engine_factory("test", first)
        with self.assertRaisesRegex(RuntimeError, "already registered"):
            engines.register_mooncake_transfer_engine_factory("test", Mock())
        engines.init_mooncake_transfer_engine("host")
        with self.assertRaisesRegex(RuntimeError, "after initialization"):
            engines.register_mooncake_transfer_engine_factory("test", Mock())
        engines.register_mooncake_transfer_engine_factory("test", first)
        self.assertIs(engines.get_mooncake_transfer_engine(), first.return_value)

    def test_unselected_plugin_preserves_native_construction(self):
        os.environ["SGLANG_MOONCAKE_TRANSFER_ENGINE"] = "mooncake"
        with patch.object(engines, "MooncakeTransferEngine") as native:
            result = engines.init_mooncake_transfer_engine("host")
        self.assertIs(result, native.return_value)
        native.assert_called_once_with(hostname="host", gpu_id=None, ib_device=None)

    def test_failed_construction_does_not_publish_a_shared_instance(self):
        factory = Mock(side_effect=RuntimeError("cannot initialize"))
        engines.register_mooncake_transfer_engine_factory("test", factory)
        with self.assertRaisesRegex(RuntimeError, "cannot initialize"):
            engines.init_mooncake_transfer_engine("host")
        self.assertIsNone(engines.get_mooncake_transfer_engine())

    def test_missing_selected_plugin_does_not_fall_back_to_native_engine(self):
        with patch.object(engines, "MooncakeTransferEngine") as native:
            with self.assertRaisesRegex(ValueError, "not registered"):
                engines.init_mooncake_transfer_engine("host")
        native.assert_not_called()

    def test_selection_cannot_change_after_initialization(self):
        engines.register_mooncake_transfer_engine_factory("test", Mock())
        first = engines.init_mooncake_transfer_engine("host")
        os.environ["SGLANG_MOONCAKE_TRANSFER_ENGINE"] = "mooncake"
        with self.assertRaisesRegex(RuntimeError, "after initialization"):
            engines.init_mooncake_transfer_engine("host")
        self.assertIs(engines.get_mooncake_transfer_engine(), first)


if __name__ == "__main__":
    unittest.main()
