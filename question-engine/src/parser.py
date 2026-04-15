from __future__ import annotations

import re

from src.llm import LlmClient
from src.models import Answer, Question


_SKIP_SENTINEL = "__SKIP__"

_FILLER_TOKENS = {"ok", "idk", "na", "nope", "dunno", "no idea", "whatever"}

_WORD_TO_DIGIT = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

_RATING_DIGIT_RE = re.compile(r"\b([1-5])(?:\s*/\s*5)?\b")
_RATING_WORD_RE = re.compile(
    r"\b(" + "|".join(_WORD_TO_DIGIT.keys()) + r")\b", re.IGNORECASE
)

_YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|affirmative|correct|true)\b", re.IGNORECASE
)
_NO_RE = re.compile(r"\b(no|nope|nah|negative|false)\b", re.IGNORECASE)


def _answer(question: Question, answer_text: str, parsed_value, status: str) -> Answer:
    return Answer(
        field_id=question.field_id,
        question_text=question.question_text,
        answer_text=answer_text,
        parsed_value=parsed_value,
        status=status,
    )


def _parse_rating(
    question: Question, answer_text: str, llm: LlmClient, model: str
) -> Answer:
    # Regex first: a bare digit 1-5, optionally followed by "/5".
    if m := _RATING_DIGIT_RE.search(answer_text):
        return _answer(question, answer_text, int(m.group(1)), "scored")
    # Word numbers ("five stars").
    if m := _RATING_WORD_RE.search(answer_text):
        return _answer(
            question, answer_text, _WORD_TO_DIGIT[m.group(1).lower()], "scored"
        )
    # LLM fallback.
    system = (
        "Extract a 1-5 rating from the guest's answer. Return JSON: "
        '{"rating": 1-5 or null, "abstain": true/false}. '
        "Set abstain=true if the answer is evasive or off-topic."
    )
    user = f"Question: {question.question_text}\nAnswer: {answer_text}"
    result = llm.chat_json(system=system, user=user, model=model)
    rating = result.get("rating")
    if isinstance(rating, int) and 1 <= rating <= 5 and not result.get("abstain"):
        return _answer(question, answer_text, rating, "scored")
    return _answer(question, answer_text, None, "unscorable")


def _parse_yes_no(
    question: Question, answer_text: str, llm: LlmClient, model: str
) -> Answer:
    yes_match = _YES_RE.search(answer_text)
    no_match = _NO_RE.search(answer_text)
    if yes_match and no_match:
        # Prefer whichever appears first.
        parsed = "yes" if yes_match.start() < no_match.start() else "no"
        return _answer(question, answer_text, parsed, "scored")
    if yes_match:
        return _answer(question, answer_text, "yes", "scored")
    if no_match:
        return _answer(question, answer_text, "no", "scored")
    # LLM fallback.
    system = (
        "Classify the guest's answer as yes/no or abstain. Return JSON: "
        '{"answer": "yes"|"no"|null, "abstain": true/false}.'
    )
    user = f"Question: {question.question_text}\nAnswer: {answer_text}"
    result = llm.chat_json(system=system, user=user, model=model)
    answer_val = result.get("answer")
    if answer_val in ("yes", "no") and not result.get("abstain"):
        return _answer(question, answer_text, answer_val, "scored")
    return _answer(question, answer_text, None, "unscorable")


def _parse_short_text(
    question: Question, answer_text: str, llm: LlmClient, model: str
) -> Answer:
    system = (
        "Extract the factual content of the guest's answer in 20 words or fewer. "
        "If the answer is vague or off-topic, set abstain=true. Return JSON: "
        '{"value": string or null, "abstain": true/false}.'
    )
    user = f"Question: {question.question_text}\nAnswer: {answer_text}"
    result = llm.chat_json(system=system, user=user, model=model)
    value = result.get("value")
    if result.get("abstain") or value is None:
        return _answer(question, answer_text, None, "unscorable")
    return _answer(question, answer_text, str(value), "scored")


def parse_answer(
    question: Question,
    answer_text: str | None,
    llm: LlmClient,
    model: str = "gpt-4.1-mini",
) -> Answer:
    """Parse a traveler's free-form answer into a structured Answer.

    Uses regex shortcuts for rating/yes-no formats; falls back to the LLM
    when the pattern is ambiguous. Short-text answers always call the LLM.
    """
    # 1. Explicit skip (None or sentinel).
    if answer_text is None or answer_text == _SKIP_SENTINEL:
        return _answer(question, answer_text or "", None, "skipped")

    # 2. Empty / trivially short — no LLM call.
    stripped = answer_text.strip()
    if len(stripped) < 3:
        return _answer(question, answer_text, None, "unscorable")

    # 3. Filler tokens — no LLM call.
    if stripped.lower() in _FILLER_TOKENS:
        return _answer(question, answer_text, None, "unscorable")

    # 4. Dispatch on input_type.
    if question.input_type == "rating_1_5":
        return _parse_rating(question, answer_text, llm, model)
    if question.input_type == "yes_no":
        return _parse_yes_no(question, answer_text, llm, model)
    if question.input_type == "short_text":
        return _parse_short_text(question, answer_text, llm, model)

    # Unknown input_type — be defensive.
    return _answer(question, answer_text, None, "unscorable")
