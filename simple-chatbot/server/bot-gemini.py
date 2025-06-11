#!/Users/donyin/miniconda3/envs/gemini-api/bin/python
#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Bot Implementation.

This module implements a chatbot using Google's Gemini Multimodal Live model.
It includes:
- Real-time audio/video interaction through Daily
- Animated robot avatar
- Speech-to-speech model

The bot runs as part of a pipeline that processes audio/video frames and manages
the conversation flow using Gemini's streaming capabilities.
"""


import asyncio
import os
import sys
from pathlib import Path
import sys

sys.path.append(str(Path().cwd()))

# ------- this is amended by Annika --------
sys.path.append(str(Path().cwd()))
with open(Path().cwd().parent.parent / "PROMPT.MD", "r") as f:
    PROMPT = f.read()

# -------- end of amendment --------

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from runner import configure

# import os

# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
# os.environ["HTTP_PROXYS"] = "http://127.0.0.1:7897"
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame, OutputImageRawFrame, SpriteFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService, InputParams, GeminiVADParams
from pipecat.transports.services.daily import DailyParams, DailyTransport

from pipecat.services.gemini_multimodal_live.events import StartSensitivity, EndSensitivity

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

sprites = []
script_dir = os.path.dirname(__file__)

for i in range(1, 26):
    full_path = os.path.join(script_dir, f"assets/robot0{i}.png")
    with Image.open(full_path) as img:
        sprites.append(OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))

# Create a smooth animation by adding reversed frames
flipped = sprites[::-1]
sprites.extend(flipped)

# Define static and animated states
quiet_frame = sprites[0]  # Static frame for when bot is listening
talking_frame = SpriteFrame(images=sprites)  # Animation sequence for when bot is talking


def load_session_memory() -> str:  # this is written by me, annika
    """this is only loaded if vtt file exists"""
    session_id = os.getenv("SESSION_ID")
    memory_dir = Path().cwd().parent.parent / "transcript" / f"{session_id}.vtt"
    if not memory_dir.exists():
        return ""
    with open(memory_dir, "r") as f:
        memory = f.read()
    return memory


class TalkingAnimation(FrameProcessor):
    """Manages the bot's visual animation states.

    Switches between static (listening) and animated (talking) states based on
    the bot's current speaking status.
    """

    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and update animation state.

        Args:
            frame: The incoming frame to process
            direction: The direction of frame flow in the pipeline
        """
        await super().process_frame(frame, direction)

        # Switch to talking animation when bot starts speaking
        if isinstance(frame, BotStartedSpeakingFrame):
            if not self._is_talking:
                await self.push_frame(talking_frame)
                self._is_talking = True
        # Return to static frame when bot stops speaking
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.push_frame(quiet_frame)
            self._is_talking = False

        await self.push_frame(frame, direction)


async def main():
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Daily video transport with specific audio parameters
    - Gemini Live multimodal model integration
    - Voice activity detection
    - Animation processing
    - RTVI event handling
    """
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        # Set up Daily transport with specific audio/video parameters for Gemini
        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=576,
                vad_enabled=True,
                vad_audio_passthrough=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            ),
        )

        # Initialize the Gemini Multimodal Live model
        llm = GeminiMultimodalLiveLLMService(
            api_key=os.getenv("GEMINI_API_KEY"),
            voice_id="Puck",  # Aoede, Charon, Fenrir, Kore, Puck
            transcribe_user_audio=True,
            enable_session_resumption=True,
            params=InputParams(
                vad=GeminiVADParams(
                    start_sensitivity=StartSensitivity.HIGH,
                    # end_sensitivity=EndSensitivity.LOW,
                    # prefix_padding_ms=300,
                    silence_duration_ms=150,
                )
            ),
        )

        memory = load_session_memory()
        if memory:
            MEMORY_PROMPT = f">>> the session has previously happened, here is the memory: {memory}; resume from where it left off"
        else:
            MEMORY_PROMPT = ""

        messages = [{"role": "user", "content": PROMPT + MEMORY_PROMPT}]

        # Set up conversation context and management
        # The context_aggregator will automatically collect conversation context
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        ta = TalkingAnimation()

        #
        # RTVI events for Pipecat client UI
        #
        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        transcript = TranscriptProcessor()

        pipeline = Pipeline(
            [
                transport.input(),
                rtvi,
                transcript.user(),
                context_aggregator.user(),
                llm,
                ta,
                transport.output(),
                transcript.assistant(),
                context_aggregator.assistant(),
            ]
        )

        # Register event handler for transcript updates
        @transcript.event_handler("on_transcript_update")
        async def handle_update(processor, frame):
            for msg in frame.messages:
                print("[Transcription:bot] ", msg.content)

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                idle_timeout_secs=0,
            ),
            observers=[RTVIObserver(rtvi)],
        )
        await task.queue_frame(quiet_frame)

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            await rtvi.set_bot_ready()

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            print(f"Participant left: {participant}")
            await task.cancel()

        runner = PipelineRunner()

        await runner.run(task)


async def run_forever():
    """Run `main()` repeatedly, whatever exit reason it has (quota, Ctrl-C, error)."""
    session_id = os.getenv("SESSION_ID", "no-session")
    while True:
        try:
            await main()  # returns when the pipeline finishes or errors
            print(f"[{session_id}] Pipeline ended - restarting in 2 s …")
        except Exception as e:
            print(f"[{session_id}] Pipeline crashed: {e} - restarting in 2 s …")
        await asyncio.sleep(2)  # small back‑off between runs


if __name__ == "__main__":
    asyncio.run(run_forever())
