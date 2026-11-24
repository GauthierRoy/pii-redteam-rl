import json
import unittest
from unittest import mock

from pii_redteam.training.llm_attack import Board, Pacer, UsageMeter, Writer, metered, parse_texts


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def reply(content: str, finish: str = "stop", usage=(100, 50)) -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
    }


class UsageMeterTest(unittest.TestCase):
    def test_tokens_and_cost(self) -> None:
        meter = UsageMeter({"w": [1.0, 4.0]})
        meter.record("w", {"prompt_tokens": 1_000_000, "completion_tokens": 500_000})
        meter.record("w", None)
        self.assertAlmostEqual(meter.cost("w"), 3.0)
        summary = meter.summary()["w"]
        self.assertEqual(summary["calls"], 2)
        self.assertEqual(summary["calls_without_usage"], 1)

    def test_metered_post_records_and_paces(self) -> None:
        clock = FakeClock()
        pacer = Pacer(0.5, sleep=clock.sleep, clock=clock)
        meter = UsageMeter({"w": [1.0, 1.0]})
        call = metered(lambda body: reply("x"), meter, pacer)
        for _ in range(3):
            call({"model": "w"})
        self.assertEqual(meter.prompt_tokens["w"], 300)
        self.assertEqual(clock.now, 1.0)  # starts at 0, 0.5, 1.0


class ParseTextsTest(unittest.TestCase):
    def test_parse(self) -> None:
        self.assertEqual(parse_texts('ok {"texts": ["a", "b"]} done'), ["a", "b"])
        self.assertIsNone(parse_texts('{"texts": "a"}'))
        self.assertIsNone(parse_texts("no json"))


class WriterTest(unittest.TestCase):
    def test_pads_and_retries(self) -> None:
        answers = iter([reply("", finish="length"), reply(json.dumps({"texts": ["one"]}))])
        writer = Writer({"model": "w", "max_tokens": 10, "attempts": 3}, lambda body: next(answers))
        with mock.patch("pii_redteam.training.llm_attack.time.sleep"):
            self.assertEqual(writer.write("p", 3), ["one", "", ""])
        self.assertEqual(writer.failed_calls, 0)

    def test_gives_up(self) -> None:
        writer = Writer({"model": "w", "max_tokens": 10, "attempts": 2}, lambda body: reply("nope"))
        with mock.patch("pii_redteam.training.llm_attack.time.sleep"):
            self.assertEqual(writer.write("p", 2), ["", ""])
        self.assertEqual(writer.failed_calls, 1)


class BoardTest(unittest.TestCase):
    def test_render(self) -> None:
        board = Board(hardest=1, easiest=1, rejected=1, max_chars=5)
        self.assertIn("first round", board.render())
        row = {"context": "email", "name": "Ana"}
        board.add(
            [
                {**row, "text": "Ana low score", "status": "valid", "p": 0.2, "reward": 1.0},
                {**row, "text": "Ana high", "status": "valid", "p": 0.99, "reward": 0.5},
                {
                    **row,
                    "text": "Ana bad",
                    "status": "invalid:name_not_person",
                    "p": None,
                    "reward": -1,
                },
            ]
        )
        text = board.render()
        self.assertIn("1 missed", text)
        self.assertIn('"Ana l..."', text)
        self.assertIn("reason=invalid:name_not_person", text)


if __name__ == "__main__":
    unittest.main()
