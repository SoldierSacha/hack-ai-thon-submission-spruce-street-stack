from langdetect import detect, DetectorFactory, LangDetectException

from src.llm import LlmClient
from src.models import TaxonomyTopic

DetectorFactory.seed = 0  # deterministic


def detect_language(text: str | None) -> str:
    if not text or not text.strip():
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def translate_to_english(
    text: str | None,
    lang: str,
    llm: LlmClient,
    model: str = "gpt-4.1-mini",
) -> str | None:
    if text is None or not text.strip():
        return text
    if lang in ("en", "unknown"):
        return text
    return llm.chat_text(
        system="Translate to English. Preserve meaning. Do not add commentary.",
        user=text,
        model=model,
        temperature=0.0,
    )


def _build_tag_prompt(topics: list[TaxonomyTopic]) -> str:
    topic_lines = "\n".join(f"- {t.topic_id}: {t.question_hint}" for t in topics)
    return (
        "You classify a hotel guest review against a fixed hospitality taxonomy.\n\n"
        "For each topic below, determine:\n"
        "- mentioned: true if the review discusses this topic, false otherwise\n"
        "- sentiment: -1 (negative), 0 (neutral), 1 (positive), or null if mentioned but tone is unclear\n"
        "- assertion: if the review makes a concrete factual claim about this topic, "
        "quote or paraphrase it in 15 words or fewer; otherwise null\n\n"
        "Return ONLY a valid JSON object mapping topic_id to {mentioned, sentiment, assertion}. "
        "Do not include topics the review does not discuss. No prose, no explanation.\n\n"
        "Topics:\n" + topic_lines
    )


def tag_review(
    review_en: str | None,
    topics: list[TaxonomyTopic],
    llm: LlmClient,
    model: str = "gpt-4.1-mini",
) -> dict[str, dict]:
    default = {"mentioned": False, "sentiment": None, "assertion": None}
    if review_en is None or not review_en.strip():
        return {t.topic_id: dict(default) for t in topics}

    raw = llm.chat_json(
        system=_build_tag_prompt(topics),
        user=review_en,
        model=model,
        temperature=0.0,
    )

    out: dict[str, dict] = {}
    for t in topics:
        entry = raw.get(t.topic_id) if isinstance(raw, dict) else None
        if not isinstance(entry, dict):
            out[t.topic_id] = dict(default)
            continue

        mentioned = bool(entry.get("mentioned", False))

        sentiment = entry.get("sentiment", None)
        if isinstance(sentiment, bool) or not isinstance(sentiment, int) or sentiment not in (-1, 0, 1):
            sentiment = None

        assertion = entry.get("assertion", None)
        if not isinstance(assertion, str) or assertion == "":
            assertion = None

        out[t.topic_id] = {
            "mentioned": mentioned,
            "sentiment": sentiment,
            "assertion": assertion,
        }
    return out
