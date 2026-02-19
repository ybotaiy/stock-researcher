"""
recommend.py - Signal generation and recommendation layer.

Two strategies:
  1. MomentumStrategy  – fully deterministic, rule-based baseline.
  2. LLMStrategy       – uses Claude to reason over evidence_pack fields.

Both return a recommendation dict with keys:
    signal   : "BUY" | "SELL" | "HOLD"
    rationale: str
    strategy : str
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
load_dotenv()

SIGNAL_CHOICES = ("BUY", "SELL", "HOLD")


# ---------------------------------------------------------------------------
# Baseline: momentum strategy
# ---------------------------------------------------------------------------

class MomentumStrategy:
    """
    Simple cross-signal momentum:
      - BUY  if ret_5d > thresh AND price > MA20
      - SELL if ret_5d < -thresh AND price < MA20
      - HOLD otherwise

    Vol filter: skip (HOLD) if 20d volatility > vol_cap.
    """

    def __init__(self, threshold: float = 0.02, vol_cap: float = 0.6):
        self.threshold = threshold
        self.vol_cap = vol_cap

    def recommend(self, evidence: dict[str, Any]) -> dict[str, Any]:
        ret5 = evidence.get("ret_5d")
        p2ma20 = evidence.get("price_to_ma20")
        vol = evidence.get("vol_20d")

        # Insufficient data guard
        if ret5 is None or p2ma20 is None:
            return {
                "signal": "HOLD",
                "rationale": "Insufficient history for signal.",
                "strategy": "momentum",
            }

        # Volatility filter
        if vol is not None and vol > self.vol_cap:
            return {
                "signal": "HOLD",
                "rationale": f"Volatility {vol:.2%} exceeds cap {self.vol_cap:.0%}; no trade.",
                "strategy": "momentum",
            }

        if ret5 > self.threshold and p2ma20 > 0:
            signal = "BUY"
            rationale = (
                f"5d return {ret5:.2%} > threshold {self.threshold:.2%} "
                f"and price {p2ma20:.2%} above MA20 → bullish momentum."
            )
        elif ret5 < -self.threshold and p2ma20 < 0:
            signal = "SELL"
            rationale = (
                f"5d return {ret5:.2%} < -{self.threshold:.2%} "
                f"and price {p2ma20:.2%} below MA20 → bearish momentum."
            )
        else:
            signal = "HOLD"
            rationale = (
                f"Mixed signals: 5d return {ret5:.2%}, price_to_ma20 {p2ma20:.2%}."
            )

        return {"signal": signal, "rationale": rationale, "strategy": "momentum"}


# ---------------------------------------------------------------------------
# LLM strategy
# ---------------------------------------------------------------------------

# Pricing in $ per token, keyed by model prefix
_PRICING_TABLE: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {
        "input":       0.80 / 1_000_000,
        "output":      4.00 / 1_000_000,
        "cache_write": 1.00 / 1_000_000,
        "cache_read":  0.08 / 1_000_000,
    },
    "claude-sonnet-4-6": {
        "input":       3.00 / 1_000_000,
        "output":     15.00 / 1_000_000,
        "cache_write": 3.75 / 1_000_000,
        "cache_read":  0.30 / 1_000_000,
    },
}
_PRICING_DEFAULT = _PRICING_TABLE["claude-sonnet-4-6"]


def _pricing_for(model: str) -> dict[str, float]:
    for prefix, rates in _PRICING_TABLE.items():
        if model.startswith(prefix):
            return rates
    return _PRICING_DEFAULT

_SYSTEM_PROMPT = """You are a quantitative equity analyst.
You will receive a JSON evidence pack for a single stock and must return a trading signal.

Rules:
- Use ONLY the fields provided in the evidence pack; do not reference external knowledge.
- Signal must be exactly one of: BUY, SELL, HOLD.
- Provide a concise rationale (1-3 sentences) citing specific field values.
- Respond with valid JSON only, no markdown fences.

Response schema:
{
  "signal": "BUY" | "SELL" | "HOLD",
  "rationale": "..."
}
"""


class LLMStrategy:
    """Uses Claude to reason over evidence_pack fields and emit a signal."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model = model
        self._client = None  # lazy init

    def _get_client(self):
        if self._client is None:
            import anthropic  # noqa: PLC0415
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        return self._client

    def recommend(self, evidence: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()

        user_msg = f"Evidence pack:\n{json.dumps(evidence, indent=2)}"

        message = client.messages.create(
            model=self.model,
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = message.content[0].text.strip()

        # Token usage + cost
        usage = message.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        pricing = _pricing_for(self.model)
        estimated_cost_usd = round(
            input_tokens * pricing["input"]
            + output_tokens * pricing["output"]
            + cache_write * pricing["cache_write"]
            + cache_read * pricing["cache_read"],
            6,
        )

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: scan for signal keyword
            signal = "HOLD"
            for s in SIGNAL_CHOICES:
                if s in raw.upper():
                    signal = s
                    break
            parsed = {"signal": signal, "rationale": raw[:300]}

        signal = parsed.get("signal", "HOLD").upper()
        if signal not in SIGNAL_CHOICES:
            signal = "HOLD"

        return {
            "signal": signal,
            "rationale": parsed.get("rationale", ""),
            "strategy": "llm",
            "model": self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": estimated_cost_usd,
        }


# ---------------------------------------------------------------------------
# Evidence pack builder
# ---------------------------------------------------------------------------

def build_evidence_pack(ticker: str, features: dict[str, Any]) -> dict[str, Any]:
    """Wrap features in a labelled evidence pack ready for JSON serialisation."""
    return {"ticker": ticker, **features}
