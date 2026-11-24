import unittest

from pii_redteam.training.gpu_cooldown import GpuCooldown, cooldown_from_config


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def make(readings, **kwargs):
    clock = FakeClock()
    feed = iter(readings)
    cooldown = GpuCooldown(
        reader=lambda: next(feed),
        sleep=clock.sleep,
        clock=clock,
        background=False,
        poll_seconds=1.0,
        **kwargs,
    )
    return cooldown, clock


class GpuCooldownTest(unittest.TestCase):
    def test_no_pause_when_cool(self) -> None:
        cooldown, _ = make([(60, 0x0)])
        cooldown.sample()
        self.assertEqual(cooldown.cool_down(), 0.0)
        self.assertEqual(cooldown.pauses, 0)

    def test_thermal_flag_triggers_pause_until_cool(self) -> None:
        # step sample: 65 C but software-thermal flag; then while paused: hot, hot, cool
        cooldown, _ = make(
            [(65, 0x20), (60, 0x0), (55, 0x0), (48, 0x0)], min_pause_seconds=2, resume_temp_c=50
        )
        cooldown.sample()
        paused = cooldown.cool_down()
        self.assertEqual(paused, 4.0)  # 2 s minimum, then two more polls until 48 C
        self.assertEqual(cooldown.pauses, 1)
        self.assertFalse(cooldown.hot_seen.is_set())

    def test_temperature_threshold_and_max_pause(self) -> None:
        cooldown, _ = make(
            [(72, 0x0)] + [(71, 0x0)] * 100, min_pause_seconds=1, max_pause_seconds=5
        )
        cooldown.sample()
        self.assertEqual(cooldown.cool_down(), 5.0)
        self.assertEqual(cooldown.summary()["max_core_temp_c"], 72)

    def test_failed_read_is_counted_not_hot(self) -> None:
        cooldown, _ = make([None])
        cooldown.sample()
        self.assertEqual(cooldown.read_failures, 1)
        self.assertEqual(cooldown.cool_down(), 0.0)

    def test_config(self) -> None:
        self.assertIsNone(cooldown_from_config({}))
        cooldown = cooldown_from_config({"gpu_cooldown": {"enabled": True, "max_temp_c": 68}})
        self.assertEqual(cooldown.max_temp_c, 68)
        cooldown.close()


if __name__ == "__main__":
    unittest.main()
