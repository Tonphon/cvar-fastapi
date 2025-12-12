import json
import re
from typing import Dict, Any
from openai import OpenAI
from ..settings import settings

def clamp_text(s: str) -> str:
    if len(s) <= settings.LLM_MAX_INPUT_CHARS:
        return s
    return s[: settings.LLM_MAX_INPUT_CHARS] + "\n\n[TRUNCATED]\n"

def _strip_code_fences(text: str) -> str:
    # remove ```json ... ``` or ``` ... ```
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def _extract_json_object(text: str) -> str:
    """
    Best-effort: find the first {...} block.
    """
    text = _strip_code_fences(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text

def generate_memo(payload: Dict[str, Any], model: str | None = None) -> Dict[str, Any]:
    client = OpenAI()
    model = model or settings.DEFAULT_MODEL

    prompt = f"""
You are a quant risk assistant. Explain results clearly for a beginner.

Return ONLY a JSON object (no markdown, no code fences) with keys:
headline, key_findings (list), risk_story, return_story, crash_days_commentary, limitations (list), next_experiments (list).

Use only the provided data (no external facts). Be concise and practical.

DATA:
{json.dumps(payload, indent=2, default=str)}
"""
    prompt = clamp_text(prompt)

    resp = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
    )

    text = (resp.output_text or "").strip()
    candidate = _extract_json_object(text)

    try:
        return json.loads(candidate)
    except Exception:
        return {
            "headline": "Quant Risk Assistant Memo",
            "key_findings": [],
            "risk_story": "",
            "return_story": "",
            "crash_days_commentary": "",
            "limitations": ["LLM output could not be parsed as JSON."],
            "next_experiments": [],
            "raw_text": text
        }
