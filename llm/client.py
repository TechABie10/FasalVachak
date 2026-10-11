import os
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI

load_dotenv()

if not os.getenv("SARVAM_API_KEY"):
    raise SystemExit(
        "SARVAM_API_KEY is not set. Add a line `SARVAM_API_KEY=<your key>` to the .env "
        "file in the project root (get a key from https://dashboard.sarvam.ai)."
    )

sarvam_client = AsyncSarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_KEY")
)

# Sarvam chat models.
# "sarvam-105b-conversations" is tuned for real-time / voice-agent workloads;
# "sarvam-105b" is the full reasoning model.
FAST_MODEL = os.getenv("SARVAM_FAST_MODEL", "sarvam-105b-conversations")
REASONING_MODEL = os.getenv("SARVAM_REASONING_MODEL", "sarvam-105b")
# Off by default: on a live call, even "low" effort took 4-13s per turn and
# often spent the whole token budget on hidden reasoning, returning no answer.
# Set to low / high / max to enable; "none" (or empty) disables it.
_effort = os.getenv("SARVAM_REASONING_EFFORT", "none").strip().lower()
REASONING_EFFORT = None if _effort in ("", "none", "off") else _effort
# Room for hidden reasoning tokens plus the answer when reasoning is enabled
# (the API default of 2048 is easily exhausted by reasoning alone).
REASONING_MAX_TOKENS = 8192


async def get_fast_llm_completion(prompt: str, temperature: float = 0.0) -> str:
    """
    Used for Phase 0 (Crop Classifier), Phase 2 (Management) and Phase 3
    (Yes/No Classifier + Follow-up QA).
    Reasoning is disabled (reasoning_effort=None) for minimum latency.
    """
    response = await sarvam_client.chat.completions(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model=FAST_MODEL,
        temperature=temperature,
        reasoning_effort=None,
    )
    return response.choices[0].message.content or ""


async def get_reasoning_completion(prompt: str, temperature: float = 0.2) -> str:
    response = await sarvam_client.chat.completions(
        messages=[
            {"role": "system", "content": "You are a helpful agricultural assistant."},
            {"role": "user", "content": prompt}
        ],
        model=REASONING_MODEL,
        temperature=temperature,
        reasoning_effort=REASONING_EFFORT,
        **({"max_tokens": REASONING_MAX_TOKENS} if REASONING_EFFORT else {}),
    )
    return response.choices[0].message.content or ""
