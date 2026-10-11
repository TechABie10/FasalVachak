# server.py
"""
FasalVachak — LiveKit Voice Agent Entry Point
=============================================

Pipeline per call
-----------------
  Farmer audio  ──► Sarvam STT (Saaras) ──► DialogueManager (our custom state
                                            machine + RAG + Sarvam LLM)
                                                    │
                                                    ▼
                                Sarvam TTS (Bulbul) ──► LiveKit room (audio track)

Key design decisions
--------------------
* We bypass VoicePipelineAgent because our LLM step is not a standard
  chat-completion call — it goes through a multi-phase dialogue state
  machine.  We wire STT → custom logic → TTS manually instead.
* A single asyncio.Lock (_turn_lock) ensures that if the farmer keeps
  talking while the agent is still generating a response, we process
  utterances one at a time (no overlapping process_turn calls that would
  corrupt DialogueManager state).
* The audio track is published once per call via tts.sarvam_client and
  reused for every response.
* On first start we play a fixed greeting immediately (no LLM, zero
  latency) — same as the original design.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.stt import SpeechEventType
from livekit.plugins import sarvam

from dialogue.session import GENERIC_ERROR_REPLY, GREETING, DialogueManager
from tts.sarvam_client import play_audio

load_dotenv()

# Logs contain Devanagari; on Windows a redirected/piped stdout defaults to the
# ANSI codepage and print() would raise mid-turn. Force UTF-8 instead.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Entrypoint — called once per incoming call
# ---------------------------------------------------------------------------

async def entrypoint(ctx: JobContext) -> None:
    """Bootstraps the agent when a farmer joins the LiveKit room."""
    await ctx.connect()
    print("[Server] Agent joined the room.")

    # Per-call state
    session = DialogueManager()

    # Serialises LLM + TTS work so concurrent STT finals never corrupt state.
    turn_lock = asyncio.Lock()

    # Strong references to background tasks — asyncio only keeps weak ones,
    # so un-referenced tasks can be garbage-collected mid-call.
    tasks: set[asyncio.Task] = set()
    # Track SIDs that already have an STT loop, so no track is processed twice.
    listening_sids: set[str] = set()

    # ── STT setup ───────────────────────────────────────────────────────────
    # Sarvam's streaming STT runs its own server-side VAD and emits one
    # FINAL_TRANSCRIPT per utterance, so no local VAD is needed.
    stt = sarvam.STT(
        language="hi-IN",           # primary language is Hindi
        model=os.getenv("SARVAM_STT_MODEL", "saaras:v4"),
        mode="transcribe",          # native-script Hindi transcript
    )

    def _start_listening(track: rtc.Track, participant: rtc.RemoteParticipant) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO or track.sid in listening_sids:
            return
        listening_sids.add(track.sid)
        print(f"[Server] Audio track subscribed from {participant.identity}.")
        task = asyncio.create_task(
            _run_stt_loop(track, ctx.room, stt, session, turn_lock)
        )
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    # ── Listen for the farmer audio track ───────────────────────────────────
    # Registered BEFORE the greeting is played: the farmer's track is often
    # subscribed while the greeting is still being synthesized, and an event
    # fired before the handler exists would be lost.
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        _start_listening(track, participant)

    # Pick up any audio tracks that were subscribed before the handler existed.
    for participant in ctx.room.remote_participants.values():
        for publication in participant.track_publications.values():
            if publication.track is not None:
                _start_listening(publication.track, participant)

    # ── Phase 0: play fixed greeting immediately (no LLM latency) ──────────
    try:
        await play_audio(ctx.room, GREETING)
    except Exception as exc:
        # A TTS hiccup on the greeting should not end the call — the farmer
        # can still speak and get answers.
        print(f"[Server] Greeting playback failed: {exc}")


# ---------------------------------------------------------------------------
# STT loop — runs in background for the lifetime of the call
# ---------------------------------------------------------------------------

async def _run_stt_loop(
    track: rtc.Track,
    room: rtc.Room,
    stt: sarvam.STT,
    session: DialogueManager,
    turn_lock: asyncio.Lock,
) -> None:
    """
    Pulls raw AudioFrames from the LiveKit track, pushes them through
    Sarvam STT, and on each FINAL_TRANSCRIPT routes the text to the
    DialogueManager.
    """
    audio_stream = rtc.AudioStream(track)
    stt_stream = stt.stream()

    # Push audio frames into the STT stream in a separate task so we can
    # consume STT events concurrently in this coroutine.
    async def _pump_audio() -> None:
        try:
            async for audio_event in audio_stream:
                stt_stream.push_frame(audio_event.frame)
        except Exception as exc:
            print(f"[Server] Audio pump failed: {exc}")
        finally:
            stt_stream.end_input()   # signal EOF when track ends

    pump_task = asyncio.create_task(_pump_audio())

    try:
        # Consume STT events
        async for event in stt_stream:
            if event.type != SpeechEventType.FINAL_TRANSCRIPT:
                continue

            farmer_text: str = event.alternatives[0].text.strip()
            if not farmer_text:
                continue

            print(f"[STT] Farmer said: {farmer_text!r}")

            # Stop accepting new turns once the call is logically over.
            if session.phase == "completed":
                print("[Server] Call completed — ignoring further input.")
                break

            # Serialise: wait for the previous turn (LLM + TTS) to finish
            # before starting the next one.
            async with turn_lock:
                if session.phase == "completed":
                    break
                try:
                    agent_reply = await session.process_turn(farmer_text)
                except Exception as exc:
                    print(f"[Server] process_turn raised: {exc}")
                    agent_reply = GENERIC_ERROR_REPLY

                if agent_reply:
                    try:
                        await play_audio(room, agent_reply)
                    except Exception as exc:
                        print(f"[Server] play_audio raised: {exc}")
    finally:
        pump_task.cancel()
        await stt_stream.aclose()
        await audio_stream.aclose()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
