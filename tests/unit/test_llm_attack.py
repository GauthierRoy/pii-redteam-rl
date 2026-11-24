import json
import unittest
from unittest import mock

from pii_redteam.training.llm_attack import (
    Board,
    BoardV2,
    Pacer,
    UsageMeter,
    Writer,
    metered,
    name_position,
    parse_answers,
    parse_texts,
)


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


class V2Test(unittest.TestCase):
    def test_parse_answers(self) -> None:
        answer = json.dumps({"hypothesis": "h", "answers": [["a"], ["b", "c", "d"]]})
        self.assertEqual(parse_answers(answer, 2, 2), ("h", [["a", ""], ["b", "c"]]))
        self.assertIsNone(parse_answers(answer, 3, 2))
        self.assertIsNone(parse_answers('{"answers": [["a"], [1]]}', 2, 2))
        objects = json.dumps({"answers": [[{"text": "a"}], ["b"]]})
        self.assertEqual(parse_answers(objects, 2, 1), ("", [["a"], ["b"]]))
        nested = json.dumps({"answers": [[json.dumps({"text": "a"})], ["{b"]]})
        self.assertEqual(parse_answers(nested, 2, 1), ("", [["a"], ["{b"]]))

    def test_name_position(self) -> None:
        self.assertEqual(name_position('"Bob, hi there', "Bob"), "first word")
        self.assertEqual(name_position("Hi Bob, see you at noon today", "Bob"), "first third")
        self.assertEqual(name_position("see you at noon today Bob", "Bob"), "last third")
        self.assertEqual(name_position("nothing", "Bob"), "absent")

    def test_board_v2_render(self) -> None:
        board = BoardV2(misses=5, closest=5, easiest=5, rejected=5, max_chars=50)
        row = {"context": "email", "reward": 0.5}
        board.add(
            [
                {**row, "name": "Ana", "text": "Ana, hi", "status": "valid", "p": 0.2},
                {**row, "name": "Ana", "text": "Hi Ana", "status": "valid", "p": 0.9},
                {**row, "name": "Ana", "text": "Ana again", "status": "valid", "p": 0.95},
                {**row, "name": "Bo", "text": "Bo Bo", "status": "invalid:name_dup", "p": None},
            ]
        )
        text = board.render("names first")
        self.assertIn("Every miss so far", text)
        self.assertIn("first word: 2, 1", text)
        self.assertEqual(text.count("Closest to a miss"), 1)
        self.assertIn("Your previous round", text)
        self.assertIn("rejected: invalid:name_dup", text)
        self.assertIn("Your hypothesis last round: names first", text)


if __name__ == "__main__":
    unittest.main()
