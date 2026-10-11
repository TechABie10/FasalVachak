# test.py
"""
Terminal-based test harness for DialogueManager.
Lets you type farmer input as text and see the agent's responses,
without needing STT/TTS hooked up yet.

Usage:
    python test.py
"""

import asyncio
import sys

from dialogue.session import GREETING, DialogueManager

DEBUG = True  # set to False to hide phase/translation info


def print_debug(dm: DialogueManager):
    if not DEBUG:
        return
    print(f"    [debug] phase={dm.phase} | crop={dm.crop} | disease={dm.identified_disease} | "
          f"questions_asked={dm.questions_asked} | followups_asked={dm.followups_asked}")
    if dm.translation_log:
        last_hindi, last_english = dm.translation_log[-1]
        print(f"    [debug] translated: '{last_hindi}' -> '{last_english}'")


async def main():
    dm = DialogueManager()

    print("=" * 60)
    print("KrishiSeva Dialogue Test CLI")
    print("Type your response as the farmer. Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    print()
    print(f"Agent: {GREETING}")

    while True:
        try:
            farmer_input = input("\nFarmer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nCall ended.")
            break

        if farmer_input.lower() in ("quit", "exit"):
            print("\nCall ended by user.")
            break

        if not farmer_input:
            print("Agent: कृपया कुछ बोलिए।")
            continue

        try:
            response = await dm.process_turn(farmer_input)
        except Exception as e:
            print(f"\n[ERROR] Something crashed: {e}")
            break

        print(f"\nAgent: {response}")
        print_debug(dm)

        if dm.phase == "completed":
            print(
                "\n--- Call has ended (phase=completed). Restart the script for a new call. ---")
            break


if __name__ == "__main__":
    # Output contains Devanagari; avoid UnicodeEncodeError on non-UTF-8 consoles.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.platform == "win32":
        # Avoids occasional event loop issues on Windows terminals
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
