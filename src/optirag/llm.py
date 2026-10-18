from __future__ import annotations

import time
from dataclasses import dataclass

from .config import settings, token_cost


class LLMError(RuntimeError):
    pass


@dataclass(slots=True)
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float


def complete(model: str, prompt: str, system: str | None = None, max_tokens: int | None = None) -> LLMResult:
    import litellm

    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    extra = {"reasoning_effort": settings.reasoning_effort} if settings.reasoning_effort else {}
    t0 = time.perf_counter()
    try:
        resp = litellm.completion(
            model=f"{settings.provider}/{model}",
            messages=messages,
            temperature=0,
            max_tokens=max_tokens or settings.max_output_tokens,
            num_retries=3,
            timeout=60,
            **extra,
        )
    except Exception as e:
        raise LLMError(f"{model}: {e}") from e
    usage = resp.usage
    in_tok, out_tok = int(usage.prompt_tokens or 0), int(usage.completion_tokens or 0)
    return LLMResult(
        text=(resp.choices[0].message.content or "").strip(),
        model=model,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost=token_cost(model, in_tok, out_tok),
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
