"""Optional OpenAI v1+ grammar cleanup. Off by default; pass-through when disabled."""

from __future__ import annotations

from openai import OpenAI

from gorisim.config import get_settings

_PROMPT = (
    "Aşağıdaki kelime dizisi Türkçe dilbilgisine uygun değil. "
    "Bu diziyi Türkçe dilbilgisine uygun bir cümleye dönüştür. "
    "Sadece cümleyi yaz, başka açıklama yapma."
)


def _client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.openai_api_key)


def grammar_clean(text: str) -> str:
    s = get_settings()
    if not s.use_llm or not s.openai_api_key:
        return text
    try:
        result = _client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
        )
        out = result.choices[0].message.content or ""
        return out.replace('"', "").strip()
    except Exception:
        return text
