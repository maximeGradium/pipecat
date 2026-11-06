#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for JambonzFrameSerializer wire format."""

import contextlib
import io
import json
import unittest

from loguru import logger

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    TTSStartedFrame,
)
from pipecat.serializers.jambonz import JambonzFrameSerializer

# The preamble jambonz sends before any audio, with identifiers anonymized.
PREAMBLE = json.dumps(
    {
        "sampleRate": 16000,
        "mixType": "mono",
        "callSid": "3e5d869a-0000-0000-0000-000000000000",
        "direction": "inbound",
        "from": "alice",
        "to": "1234",
        "callId": "94682bcc-0000-0000-0000-000000000000",
        "sipStatus": 200,
        "sipReason": "OK",
        "callStatus": "in-progress",
        "accountSid": "00000000-0000-0000-0000-000000000000",
        "traceId": "00000000000000000000000000000000",
    }
)

JAMBONZ_RATE = 16000
FRAME_BYTES = 640  # 20 ms of 16-bit mono at 16 kHz, the size jambonz sends


def audio(num_bytes: int) -> bytes:
    return b"\x01\x02" * (num_bytes // 2)


def bot_audio(sample_rate: int = JAMBONZ_RATE) -> OutputAudioRawFrame:
    return OutputAudioRawFrame(audio=audio(FRAME_BYTES), sample_rate=sample_rate, num_channels=1)


def preamble_with(**overrides) -> str:
    """The preamble with fields replaced, for the cases jambonz can negotiate."""
    return json.dumps({**json.loads(PREAMBLE), **overrides})


@contextlib.contextmanager
def captured_warnings():
    """Capture loguru warnings (pipecat uses loguru, not stdlib logging)."""
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        yield sink
    finally:
        logger.remove(handler_id)


async def serializer_after_preamble(preamble: str = PREAMBLE, **params) -> JambonzFrameSerializer:
    serializer = JambonzFrameSerializer(JambonzFrameSerializer.InputParams(**params))
    await serializer.deserialize(preamble)
    return serializer


class TestJambonzDisconnect(unittest.IsolatedAsyncioTestCase):
    """Ending the pipeline should end the call.

    `disconnect` ends the `listen` verb, after which jambonz runs the verb's
    actionHook or hangs up. This is the jambonz equivalent of Twilio's
    auto_hang_up, and it needs no API credentials.
    """

    async def test_the_pipeline_ending_ends_the_call(self):
        for frame in (EndFrame(), CancelFrame()):
            with self.subTest(frame=type(frame).__name__):
                serializer = await serializer_after_preamble()

                sent = await serializer.serialize(frame)

                self.assertEqual(json.loads(sent), {"type": "disconnect"})

    async def test_disconnect_is_sent_once(self):
        """Both frames can arrive on one shutdown, and the second would be
        written to a socket jambonz has already closed."""
        serializer = await serializer_after_preamble()

        self.assertIsNotNone(await serializer.serialize(EndFrame()))
        self.assertIsNone(await serializer.serialize(CancelFrame()))
        self.assertIsNone(await serializer.serialize(EndFrame()))

    async def test_disconnect_can_be_turned_off(self):
        """For an application that ends the verb itself, or wants the call to
        outlive the pipeline."""
        serializer = await serializer_after_preamble(disconnect_on_end=False)

        self.assertIsNone(await serializer.serialize(EndFrame()))
        self.assertIsNone(await serializer.serialize(CancelFrame()))


class TestJambonzTextFrames(unittest.IsolatedAsyncioTestCase):
    """The socket is not binary-only after the preamble.

    With `passDtmf` set, jambonz sends digits as text frames, and it answers
    `mark` and `killAudio` the same way. Routing all of those through the
    preamble handler replaced `call_meta` and set `call_sid` and `trace_id` to
    None, and lost the digit.
    """

    async def asyncSetUp(self):
        self.serializer = await serializer_after_preamble()
        self.call_sid = self.serializer.call_sid
        self.trace_id = self.serializer.trace_id

    def assert_call_survived(self):
        self.assertEqual(self.serializer.call_sid, self.call_sid)
        self.assertEqual(self.serializer.trace_id, self.trace_id)
        self.assertEqual(self.serializer.audio_in_sample_rate, JAMBONZ_RATE)

    async def test_a_dtmf_digit_becomes_a_frame_and_keeps_the_call(self):
        frame = await self.serializer.deserialize(
            json.dumps({"event": "dtmf", "dtmf": "5", "duration": "1600"})
        )

        self.assertIsInstance(frame, InputDTMFFrame)
        self.assertEqual(frame.button, KeypadEntry.FIVE)
        self.assert_call_survived()

    async def test_every_keypad_entry_is_recognised(self):
        for digit in "0123456789*#":
            with self.subTest(digit=digit):
                frame = await self.serializer.deserialize(
                    json.dumps({"event": "dtmf", "dtmf": digit})
                )
                self.assertIsInstance(frame, InputDTMFFrame)
                self.assertEqual(frame.button.value, digit)

    async def test_a_digit_that_is_not_on_a_keypad_is_ignored(self):
        """Dropping it beats raising inside the transport's read loop."""
        self.assertIsNone(
            await self.serializer.deserialize(json.dumps({"event": "dtmf", "dtmf": "A"}))
        )
        self.assert_call_survived()

    async def test_mark_and_play_done_replies_are_ignored(self):
        for event in (
            {"type": "mark", "data": {"name": "my-mark-1", "event": "playout"}},
            {"type": "mark", "data": {"name": "my-mark-1", "event": "cleared"}},
            {"type": "playDone"},
        ):
            with self.subTest(event=event):
                self.assertIsNone(await self.serializer.deserialize(json.dumps(event)))
                self.assert_call_survived()

    async def test_only_the_first_text_frame_is_the_preamble(self):
        """A second preamble-shaped frame is an event, not a renegotiation."""
        await self.serializer.deserialize(
            preamble_with(sampleRate=8000, callSid="other", traceId="other")
        )

        self.assert_call_survived()

    async def test_an_undecodable_event_is_ignored(self):
        self.assertIsNone(await self.serializer.deserialize("not json"))
        self.assert_call_survived()

    async def test_audio_still_arrives_after_an_event(self):
        """The regression that mattered: a digit left the rate unusable, so
        every audio frame after it was dropped."""
        await self.serializer.deserialize(json.dumps({"event": "dtmf", "dtmf": "1"}))

        self.assertIsNotNone(await self.serializer.deserialize(audio(FRAME_BYTES)))


class TestJambonzMixType(unittest.IsolatedAsyncioTestCase):
    """Only mono is interpretable; the other two layouts carry two channels."""

    async def test_a_mono_call_is_interpreted(self):
        serializer = await serializer_after_preamble(preamble_with(mixType="mono"))
        self.assertEqual(serializer.mix_type, "mono")

        self.assertIsNotNone(await serializer.deserialize(audio(FRAME_BYTES)))

    async def test_stereo_is_dropped_rather_than_misread(self):
        """Two interleaved channels read as mono play at double speed with the
        channels alternating, which is worse than no audio at all."""
        serializer = await serializer_after_preamble(preamble_with(mixType="stereo"))
        self.assertEqual(serializer.mix_type, "stereo")

        self.assertIsNone(await serializer.deserialize(audio(FRAME_BYTES)))

    async def test_mixed_is_dropped_even_though_it_would_decode(self):
        """`mixed` is a single channel, so it would be interpreted correctly.
        It is refused because it is both legs summed, so the bot would be fed
        its own audio -- a different reason from stereo, and the one that is
        easy to get wrong."""
        serializer = await serializer_after_preamble(preamble_with(mixType="mixed"))
        self.assertEqual(serializer.mix_type, "mixed")

        self.assertIsNone(await serializer.deserialize(audio(FRAME_BYTES)))


class TestJambonzDeserialize(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = await serializer_after_preamble()

    async def test_sample_rate_is_read_from_the_preamble(self):
        self.assertEqual(self.serializer.audio_in_sample_rate, JAMBONZ_RATE)

    async def test_call_identifiers_are_kept(self):
        self.assertEqual(self.serializer.call_sid, "3e5d869a-0000-0000-0000-000000000000")
        self.assertEqual(self.serializer.trace_id, "00000000000000000000000000000000")

    async def test_audio_becomes_input_frames_at_the_call_rate(self):
        """The input transport resamples to the pipeline's rate, not the serializer."""
        data = audio(FRAME_BYTES)

        frame = await self.serializer.deserialize(data)

        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.audio, data)
        self.assertEqual(frame.sample_rate, JAMBONZ_RATE)
        self.assertEqual(frame.num_channels, 1)

    async def test_an_undecodable_text_frame_is_ignored(self):
        self.assertIsNone(await self.serializer.deserialize("not json"))
        self.assertEqual(self.serializer.audio_in_sample_rate, JAMBONZ_RATE)


class TestJambonzPreambleHandling(unittest.IsolatedAsyncioTestCase):
    async def test_a_narrowband_call_is_followed(self):
        """8 kHz is valid for narrowband SIP and must need no configuration."""
        serializer = await serializer_after_preamble(preamble_with(sampleRate=8000))

        self.assertEqual(serializer.audio_in_sample_rate, 8000)
        frame = await serializer.deserialize(audio(FRAME_BYTES))
        self.assertEqual(frame.sample_rate, 8000)

    async def test_audio_before_the_preamble_is_dropped(self):
        serializer = JambonzFrameSerializer()

        self.assertIsNone(await serializer.deserialize(audio(FRAME_BYTES)))

    async def test_a_configured_rate_is_not_overridden_by_the_preamble(self):
        serializer = await serializer_after_preamble(audio_in_sample_rate=8000)

        self.assertEqual(serializer.audio_in_sample_rate, 8000)
        frame = await serializer.deserialize(audio(FRAME_BYTES))
        self.assertEqual(frame.sample_rate, 8000)

    async def test_a_text_frame_that_is_not_an_object_is_ignored(self):
        serializer = JambonzFrameSerializer()

        self.assertIsNone(await serializer.deserialize("[1, 2, 3]"))
        self.assertIsNone(serializer.audio_in_sample_rate)

    async def test_the_whole_preamble_is_available(self):
        serializer = await serializer_after_preamble()

        self.assertEqual(serializer.call_meta["sipReason"], "OK")
        # A copy, so a caller cannot edit the serializer's own record of the call.
        serializer.call_meta["sipReason"] = "edited"
        self.assertEqual(serializer.call_meta["sipReason"], "OK")

    async def test_outgoing_audio_does_not_wait_for_the_preamble(self):
        """The return path's rate isn't in the preamble, so nothing waits on it."""
        serializer = JambonzFrameSerializer()

        frame = bot_audio()
        self.assertEqual(await serializer.serialize(frame), frame.audio)


class TestJambonzSerializeWireFormat(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = await serializer_after_preamble()

    async def test_audio_is_sent_as_raw_bytes(self):
        """The transport sends at the call's rate, so the audio goes out untouched."""
        frame = bot_audio()

        sent = await self.serializer.serialize(frame)

        self.assertIsInstance(sent, bytes)
        self.assertEqual(sent, frame.audio)

    async def test_interruption_becomes_kill_audio(self):
        message = json.loads(await self.serializer.serialize(InterruptionFrame()))
        self.assertEqual(message, {"type": "killAudio"})

    async def test_transport_messages_are_forwarded_as_json(self):
        frame = OutputTransportMessageFrame(message={"type": "custom", "value": 1})

        self.assertEqual(
            json.loads(await self.serializer.serialize(frame)),
            {"type": "custom", "value": 1},
        )

    async def test_rtvi_messages_are_not_forwarded(self):
        """RTVI is a browser protocol; jambonz would read it as a command."""
        frame = OutputTransportMessageFrame(message={"label": "rtvi-ai", "type": "bot-ready"})

        self.assertIsNone(await self.serializer.serialize(frame))

    async def test_an_unhandled_frame_produces_nothing(self):
        self.assertIsNone(await self.serializer.serialize(TTSStartedFrame()))

    async def test_audio_keeps_flowing_after_kill_audio(self):
        """jambonz reads the socket on one loop, so frames sent after killAudio
        are queued rather than racing the flush, and Pipecat's transport drops
        its own queued output on an interruption. Withholding audio here would
        add latency to every barge-in recovery for no reason."""
        await self.serializer.serialize(InterruptionFrame())

        self.assertIsNotNone(await self.serializer.serialize(bot_audio()))


class TestJambonzReturnRate(unittest.IsolatedAsyncioTestCase):
    """jambonz reads the bot's audio at `bidirectionalAudio.sampleRate`, which
    the preamble doesn't state, so the serializer can only check what it's told."""

    async def test_audio_at_the_configured_rate_is_sent(self):
        serializer = await serializer_after_preamble(audio_out_sample_rate=8000)
        self.assertEqual(serializer.audio_out_sample_rate, 8000)

        with captured_warnings() as warnings:
            self.assertIsNotNone(await serializer.serialize(bot_audio(8000)))

        self.assertEqual(warnings.getvalue(), "")

    async def test_audio_off_the_configured_rate_is_dropped_with_one_warning(self):
        """Audio serialized outside a transport isn't resampled to the wire's rate."""
        serializer = await serializer_after_preamble(audio_out_sample_rate=8000)

        with captured_warnings() as warnings:
            self.assertIsNone(await serializer.serialize(bot_audio(24000)))
            self.assertIsNone(await serializer.serialize(bot_audio(24000)))

        self.assertEqual(warnings.getvalue().count("dropping 24000 Hz audio"), 1)

    async def test_audio_off_the_call_rate_warns_once_when_no_return_rate_is_set(self):
        """Most likely the pipeline's default rate against a verb set to the call's."""
        serializer = await serializer_after_preamble()

        with captured_warnings() as warnings:
            self.assertIsNotNone(await serializer.serialize(bot_audio(24000)))
            self.assertIsNotNone(await serializer.serialize(bot_audio(24000)))

        self.assertEqual(warnings.getvalue().count("Sending 24000 Hz audio"), 1)

    async def test_audio_at_the_call_rate_does_not_warn(self):
        serializer = await serializer_after_preamble()

        with captured_warnings() as warnings:
            await serializer.serialize(bot_audio())

        self.assertEqual(warnings.getvalue(), "")
