from langdetect import detect, DetectorFactory, LangDetectException

from src.llm import LlmClient

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
