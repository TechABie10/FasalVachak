# llm/translation.py
import os

from llm.client import sarvam_client

# Sarvam Translate API. "mayura:v1" handles colloquial / code-mixed spoken input;
# "sarvam-translate:v1" is the alternative.
TRANSLATION_MODEL = os.getenv("SARVAM_TRANSLATION_MODEL", "mayura:v1")


async def translate_to_english(hindi_text: str) -> str:
    response = await sarvam_client.text.translate(
        input=hindi_text,
        # "auto" so Hinglish / English words mixed into the farmer's speech still work.
        source_language_code="auto",
        target_language_code="en-IN",
        model=TRANSLATION_MODEL,
    )
    result = (response.translated_text or "").strip()
    print(f"[Translation] '{hindi_text[:60]}' → '{result[:60]}'")
    return result
