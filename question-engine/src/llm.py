import hashlib, json, os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class LlmClient:
    def __init__(self, cache_dir: str | Path = "data/cache",
                 api_key: str | None = None):
        if api_key is None:
            load_dotenv()
        self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def _key(self, **parts) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    def _cache_get(self, key):
        p = self.cache / f"{key}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _cache_put(self, key, value):
        (self.cache / f"{key}.json").write_text(json.dumps(value))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat_json(self, *, system: str, user: str, model: str,
                  temperature: float = 0.0) -> dict:
        key = self._key(kind="chat_json", system=system, user=user,
                        model=model, temperature=temperature)
        if (hit := self._cache_get(key)) is not None: return hit
        resp = self._client.chat.completions.create(
            model=model, temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        out = json.loads(resp.choices[0].message.content)
        self._cache_put(key, out)
        return out

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat_text(self, *, system: str, user: str, model: str,
                  temperature: float = 0.3) -> str:
        key = self._key(kind="chat_text", system=system, user=user,
                        model=model, temperature=temperature)
        if (hit := self._cache_get(key)) is not None: return hit["text"]
        resp = self._client.chat.completions.create(
            model=model, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        text = resp.choices[0].message.content.strip()
        self._cache_put(key, {"text": text})
        return text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed(self, text: str, model: str = "text-embedding-3-small") -> list[float]:
        key = self._key(kind="embed", text=text, model=model)
        if (hit := self._cache_get(key)) is not None: return hit["vector"]
        resp = self._client.embeddings.create(model=model, input=[text])
        vec = resp.data[0].embedding
        self._cache_put(key, {"vector": vec})
        return vec
