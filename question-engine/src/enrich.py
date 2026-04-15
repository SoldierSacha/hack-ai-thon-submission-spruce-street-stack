from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0  # deterministic


def detect_language(text: str | None) -> str:
    if not text or not text.strip():
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"
