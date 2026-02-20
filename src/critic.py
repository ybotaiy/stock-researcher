"""
critic.py - LLM critic step for recommendation review.

critique_recommendation(evidence, recommendation, model, client) -> dict

The critic reads the evidence pack and the initial recommendation, flags
logical conflicts, and may downgrade the signal to HOLD or reduce confidence.
It never increases confidence above the initial value.
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from src.recommend import SIGNAL_CHOICES, _pricing_for

_DEFAULT_CRITIC_MODEL = "claude-haiku-4-5-20251001"

_CRITIC_SYSTEM_PROMPT = """You are a senior quantitative risk analyst acting as a critic.
You will receive:
  1. A JSON evidence pack for a single stock.
  2. A preliminary trading recommendation (signal, confidence, rationale).

Your job is to critically review the recommendation against the evidence pack.

Rules:
- Use ONLY the fields provided in the evidence pack; do not reference external knowledge.
- Flag logical conflicts (e.g. a BUY signal when ret_5d is negative and price is below MA20).
- You may downgrade the signal to HOLD if the evidence contradicts the recommendation.
- You may reduce confidence if the evidence is ambiguous or inconsistent with the signal.
- You must NEVER increase confidence above the original value.
- Respond with valid JSON only, no markdown fences.

Response schema:
{
  "agree": true | false,
  "adjusted_signal": "BUY" | "SELL" | "HOLD",
  "adjusted_confidence": 0.0-1.0,
  "critique": "1-3 sentences explaining your assessment"
}
"""


def critique_recommendation(
    evidence: dict[str, Any],
    recommendation: dict[str, Any],
    model: str | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """Critique an initial recommendation against the evidence pack.

    Parameters
    ----------
    evidence       : evidence pack dict (same as passed to strategy.recommend)
    recommendation : dict returned by strategy.recommend(), must have 'signal' and 'confidence'
    model          : Claude model ID (defaults to claude-haiku-4-5-20251001)
    client         : optional pre-built anthropic.Anthropic client to reuse

    Returns
    -------
    dict with keys:
        critic_agree, confidence_before, confidence_after, stance_before,
        stance_after, critique, critic_input_tokens, critic_output_tokens,
        critic_estimated_cost_usd
    """
    if model is None:
        model = _DEFAULT_CRITIC_MODEL

    if client is None:
        import anthropic  # noqa: PLC0415
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    stance_before = recommendation.get("signal", "HOLD")
    confidence_before = float(recommendation.get("confidence", 0.5))

    user_msg = (
        f"Evidence pack:\n{json.dumps(evidence, indent=2)}\n\n"
        f"Preliminary recommendation:\n{json.dumps(recommendation, indent=2)}"
    )

    message = client.messages.create(
        model=model,
        max_tokens=256,
        system=_CRITIC_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = message.content[0].text.strip()

    usage = message.usage
    critic_input_tokens = usage.input_tokens
    critic_output_tokens = usage.output_tokens
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    pricing = _pricing_for(model)
    critic_estimated_cost_usd = round(
        critic_input_tokens * pricing["input"]
        + critic_output_tokens * pricing["output"]
        + cache_write * pricing["cache_write"]
        + cache_read * pricing["cache_read"],
        6,
    )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: agree=True, keep original signal and confidence
        return {
            "critic_agree": True,
            "confidence_before": confidence_before,
            "confidence_after": confidence_before,
            "stance_before": stance_before,
            "stance_after": stance_before,
            "critique": raw[:300],
            "critic_input_tokens": critic_input_tokens,
            "critic_output_tokens": critic_output_tokens,
            "critic_estimated_cost_usd": critic_estimated_cost_usd,
        }

    critic_agree = bool(parsed.get("agree", True))

    adjusted_signal = str(parsed.get("adjusted_signal", stance_before)).upper()
    if adjusted_signal not in SIGNAL_CHOICES:
        adjusted_signal = "HOLD"

    try:
        adjusted_confidence = float(parsed.get("adjusted_confidence", confidence_before))
        adjusted_confidence = max(0.0, min(adjusted_confidence, confidence_before))
    except (TypeError, ValueError):
        adjusted_confidence = confidence_before

    return {
        "critic_agree": critic_agree,
        "confidence_before": confidence_before,
        "confidence_after": adjusted_confidence,
        "stance_before": stance_before,
        "stance_after": adjusted_signal,
        "critique": str(parsed.get("critique", ""))[:300],
        "critic_input_tokens": critic_input_tokens,
        "critic_output_tokens": critic_output_tokens,
        "critic_estimated_cost_usd": critic_estimated_cost_usd,
    }
