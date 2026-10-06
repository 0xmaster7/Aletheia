"""Frozen intent-specific scorer for the professor-feedback evaluation."""
from __future__ import annotations

import re
from typing import Any


def normalize_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def parse_boolean(value: Any) -> bool | None:
    text = str(value).casefold()
    tokens = re.findall(r"(?<![a-z])(yes|true|correct|no|false|incorrect)(?![a-z])", text)
    if not tokens:
        return None
    return tokens[-1] in {"yes", "true", "correct"}


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}
_NUMBER_TOKEN = re.compile(r"(?<![\w])([+-]?\d+)(?![\w])|(?<![a-z])(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)(?![a-z])", re.I)
_FINAL_ANSWER = re.compile(r"(?:final\s+answer|answer)\s*[:=-]\s*(.*)$", re.I | re.S)


def parse_final_number(value: Any) -> int | None:
    text = str(value)
    markers = list(_FINAL_ANSWER.finditer(text))
    if markers:
        text = markers[-1].group(1)
    found: list[int] = []
    for match in _NUMBER_TOKEN.finditer(text.replace(",", "")):
        digits, word = match.groups()
        found.append(int(digits) if digits is not None else _NUMBER_WORDS[word.casefold()])
    return found[-1] if found else None


def score_answer(intent: str, predicted: Any, gold: Any) -> bool:
    """Score booleans canonically, counts exactly, and text with the fixed SubEM rule."""
    if predicted is None:
        return False
    if intent == "boolean":
        prediction = parse_boolean(predicted)
        expected = parse_boolean(gold)
        return prediction is not None and expected is not None and prediction is expected
    if intent == "aggregation":
        try:
            expected = int(str(gold).strip())
        except (TypeError, ValueError):
            return False
        return parse_final_number(predicted) == expected
    normalized_prediction = normalize_text(predicted)
    normalized_gold = normalize_text(gold)
    return bool(normalized_gold) and (
        normalized_prediction == normalized_gold
        or normalized_gold in normalized_prediction
    )
