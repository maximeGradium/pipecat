#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for frame-based audio handling in :class:`BaseInputTransport`."""

import asyncio
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import (
    CancelFrame,
    FilterEnableFrame,
    FilterUpdateSettingsFrame,
    InputAudioRawFrame,
    InputTransportStartAudioStreamingFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


class TestBaseInputTransportFrameAudio(unittest.IsolatedAsyncioTestCase):
    def _transport(self) -> BaseInputTransport:
        return BaseInputTransport(TransportParams(audio_in_enabled=True))

    async def test_incoming_audio_frame_routed_to_push_audio_frame(self):
        transport = self._transport()
        transport.push_audio_frame = AsyncMock()
        transport.push_frame = AsyncMock()
        frame = InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
        await transport.process_frame(frame, FrameDirection.DOWNSTREAM)
        # Fed into the VAD path, not forwarded as a plain frame.
        transport.push_audio_frame.assert_called_once_with(frame)

    async def test_start_audio_streaming_frame_triggers_streaming(self):
        transport = self._transport()
        transport._start_audio_in_streaming = AsyncMock()
        await transport.process_frame(
            InputTransportStartAudioStreamingFrame(), FrameDirection.DOWNSTREAM
        )
        transport._start_audio_in_streaming.assert_called_once()

    async def test_start_audio_in_streaming_method_is_deprecated(self):
        transport = self._transport()
        transport._start_audio_in_streaming = AsyncMock()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await transport.start_audio_in_streaming()
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
        transport._start_audio_in_streaming.assert_called_once()


class TestBaseInputTransportFilterFrames(unittest.IsolatedAsyncioTestCase):
    def _transport(self) -> BaseInputTransport:
        audio_in_filter = AsyncMock(spec=BaseAudioFilter)
        return BaseInputTransport(TransportParams(audio_in_filter=audio_in_filter))

    async def test_filter_control_frames_go_to_input_filter(self):
        for frame in (FilterEnableFrame(enable=False), FilterUpdateSettingsFrame(settings={})):
            with self.subTest(frame=frame.name):
                transport = self._transport()
                transport.push_frame = AsyncMock()
                await transport.process_frame(frame, FrameDirection.DOWNSTREAM)
                transport._params.audio_in_filter.process_frame.assert_called_once_with(frame)
                transport.push_frame.assert_not_called()


class TestBaseInputTransportResampling(unittest.IsolatedAsyncioTestCase):
    """Audio at another rate, such as a serializer's wire rate, is resampled to
    the transport's input rate before it goes down the pipeline."""

    INPUT_RATE = 16000
    WIRE_RATE = 8000

    async def asyncSetUp(self):
        self.transport = BaseInputTransport(TransportParams(audio_in_enabled=True))
        self.transport.push_frame = AsyncMock()
        await self.transport.setup(
            frame_processor_setup(
                task_manager=TaskManager(),
                pipeline_worker=SimpleNamespace(app_resources=None),
                audio_in_sample_rate=self.INPUT_RATE,
            )
        )
        start_frame = StartFrame(audio_in_sample_rate=self.INPUT_RATE)
        await self.transport.process_frame(start_frame, FrameDirection.DOWNSTREAM)
        await self.transport.set_transport_ready(start_frame)

    async def asyncTearDown(self):
        await self.transport.cancel(CancelFrame())

    async def _passed_on(self, frames: list[InputAudioRawFrame]) -> list[InputAudioRawFrame]:
        for frame in frames:
            await self.transport.push_audio_frame(frame)
        await asyncio.sleep(0.05)
        return [
            call.args[0]
            for call in self.transport.push_frame.call_args_list
            if isinstance(call.args[0], InputAudioRawFrame)
        ]

    async def test_audio_at_another_rate_is_resampled(self):
        wire_frames = [
            InputAudioRawFrame(audio=b"\x01\x02" * 160, sample_rate=self.WIRE_RATE, num_channels=1)
            for _ in range(25)
        ]

        passed_on = await self._passed_on(wire_frames)

        self.assertTrue(passed_on)
        for frame in passed_on:
            self.assertEqual(frame.sample_rate, self.INPUT_RATE)
            self.assertTrue(frame.audio, "a frame the resampler held back was passed on")
            self.assertEqual(frame.num_frames, len(frame.audio) // 2)
        received = sum(frame.num_frames for frame in passed_on)
        self.assertLessEqual(received, 25 * 160 * self.INPUT_RATE // self.WIRE_RATE)

    async def test_audio_at_the_input_rate_is_passed_on_untouched(self):
        audio = b"\x01\x02" * 320
        frame = InputAudioRawFrame(audio=audio, sample_rate=self.INPUT_RATE, num_channels=1)

        passed_on = await self._passed_on([frame])

        self.assertEqual(passed_on, [frame])
        self.assertEqual(frame.audio, audio)


if __name__ == "__main__":
    unittest.main()
