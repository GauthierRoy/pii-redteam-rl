"""Collapse guard: opening word extraction and the share across other requests."""

import unittest

from pii_redteam.training.opening_share import OpeningShare, opening_word


class TestOpeningWord(unittest.TestCase):
    def test_first_word_is_lower_cased(self):
        self.assertEqual(opening_word("Dear Lucas, your order shipped.", "Lucas"), "dear")

    def test_requested_name_at_the_start_is_skipped(self):
        self.assertEqual(opening_word("Lucas reported a broken latch.", "Lucas"), "reported")

    def test_multi_word_name_is_skipped_as_a_whole(self):
        self.assertEqual(opening_word("Ana Ruiz | stall 14 | paid", "Ana Ruiz"), "stall")

    def test_name_inside_a_longer_word_is_not_removed(self):
        self.assertEqual(opening_word("Annual report for Ann", "Ann"), "annual")

    def test_text_without_letters_has_no_opening(self):
        self.assertEqual(opening_word("Lucas: 12 / 14", "Lucas"), "")


class TestOpeningShare(unittest.TestCase):
    def guard(self, **overrides):
        settings = {"threshold": 0.2, "window": 100, "min_other_texts": 4, **overrides}
        return OpeningShare(**settings)

    def test_share_counts_only_texts_written_for_other_requests(self):
        guard = self.guard()
        requests = ["a", "a", "b", "b", "c", "c"]
        openings = ["dear", "dear", "dear", "hi", "hey", "row"]
        shares = guard.shares(requests, openings)
        # request a: the four other texts hold one "dear"; its own second "dear" is ignored
        self.assertEqual(shares[0], 0.25)
        # request b's "dear": two of the four other texts open with "dear"
        self.assertEqual(shares[2], 0.5)
        self.assertEqual(shares[4], 0.0)

    def test_one_request_repeating_its_own_opening_is_not_a_collapse(self):
        guard = self.guard()
        shares = guard.shares(["a"] * 4 + ["b"] * 4, ["dear"] * 4 + ["hi", "hey", "row", "re"])
        self.assertEqual(shares[:4], [0.0] * 4)

    def test_no_share_until_enough_other_texts_are_seen(self):
        guard = self.guard(min_other_texts=6)
        first = guard.shares(["a", "a", "b", "b"], ["dear"] * 4)
        self.assertEqual(first, [None] * 4)
        self.assertFalse(guard.is_repeated(first[0]))
        second = guard.shares(["c", "c", "d", "d"], ["dear"] * 4)
        self.assertEqual(second, [1.0] * 4)

    def test_window_forgets_old_texts(self):
        guard = self.guard(window=4, min_other_texts=2)
        guard.shares(["a", "a", "b", "b"], ["dear"] * 4)
        shares = guard.shares(["c", "c", "d", "d"], ["hi", "hey", "row", "re"])
        self.assertEqual(shares, [0.0] * 4)

    def test_threshold_is_exclusive(self):
        guard = self.guard(threshold=0.2)
        self.assertFalse(guard.is_repeated(0.2))
        self.assertTrue(guard.is_repeated(0.21))


if __name__ == "__main__":
    unittest.main()
