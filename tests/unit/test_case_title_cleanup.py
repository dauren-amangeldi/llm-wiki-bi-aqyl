"""Filename cleanup must preserve meaningful business terms and numbers."""

import unittest

from llm_wiki.case_titles import clean_automatic_case_title


class CaseTitleCleanupTests(unittest.TestCase):
    def test_file_style_names(self):
        for original, expected in [
            ("02-25_Lecture_Modern_Marketing_Strategy", "Lecture Modern Marketing Strategy"),
            ("2026-02-25_Strategic__Marketing_Leadership", "Strategic Marketing Leadership"),
            ("25.02.2026_Стратегия_BI_Group", "Стратегия BI Group"),
            ("02-25_Strategy · 02-26_Brand_Positioning", "Strategy · Brand Positioning"),
            ("02-25_Report.PDF", "Report"),
            ("02-25__", ""),
        ]:
            with self.subTest(original=original):
                self.assertEqual(clean_automatic_case_title(original), expected)

    def test_meaningful_names_and_numbers_are_preserved(self):
        for title in [
            "2025 strategy", "ISO 9001", "GPT-4 for BI Group", "Price-Value Plays",
            "260225 Final lecture", "iPhone and eBay", "Рынок Казахстана в 2026 году",
            "Период 02-25", "99-99 Market", "B2B: 3 шага",
        ]:
            with self.subTest(title=title):
                self.assertEqual(clean_automatic_case_title(title), title)
