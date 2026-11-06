#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the wire sample rates of the telephony serializers."""

import base64
import json
import unittest
import warnings

from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.serializers.exotel import ExotelFrameSerializer
from pipecat.serializers.genesys import GenesysAudioHookSerializer
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.serializers.vonage import VonageFrameSerializer
from pipecat.utils import deprecation

# Each serializer, the field it used to take its wire rate from, and that rate.
PROVIDERS = [
    (TwilioFrameSerializer, "twilio_sample_rate", 8000),
    (TelnyxFrameSerializer, "telnyx_sample_rate", 8000),
    (PlivoFrameSerializer, "plivo_sample_rate", 8000),
    (ExotelFrameSerializer, "exotel_sample_rate", 8000),
    (GenesysAudioHookSerializer, "genesys_sample_rate", 8000),
    (VonageFrameSerializer, "vonage_sample_rate", 16000),
]


def deprecations_while(build):
    """What `build` returns, and the deprecation warnings it raised."""
    deprecation._warned_sites.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = build()
    return result, [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]


class TestWireRates(unittest.TestCase):
    def test_each_serializer_states_its_wire_rate(self):
        for cls, _, rate in PROVIDERS:
            with self.subTest(serializer=cls.__name__):
                params = cls.InputParams()
                self.assertEqual(params.audio_in_sample_rate, rate)
                self.assertEqual(params.audio_out_sample_rate, rate)

    def test_a_serializer_without_a_fixed_rate_states_none(self):
        """A wire format that carries the rate leaves the transport's rate alone."""
        params = FrameSerializer.InputParams()
        self.assertIsNone(params.audio_in_sample_rate)
        self.assertIsNone(params.audio_out_sample_rate)


class TestDeprecatedRateFields(unittest.TestCase):
    def test_the_provider_rate_sets_both_wire_rates(self):
        for cls, field, _ in PROVIDERS:
            with self.subTest(serializer=cls.__name__):
                params, messages = deprecations_while(
                    lambda cls=cls, field=field: cls.InputParams(**{field: 16000})
                )

                self.assertEqual(params.audio_in_sample_rate, 16000)
                self.assertEqual(params.audio_out_sample_rate, 16000)
                self.assertTrue([m for m in messages if f".InputParams.{field}`" in m])

    def test_a_wire_rate_given_alongside_wins(self):
        params, _ = deprecations_while(
            lambda: TwilioFrameSerializer.InputParams(
                twilio_sample_rate=16000, audio_out_sample_rate=8000
            )
        )

        self.assertEqual(params.audio_in_sample_rate, 16000)
        self.assertEqual(params.audio_out_sample_rate, 8000)

    def test_sample_rate_warns(self):
        for cls, _, _ in PROVIDERS:
            with self.subTest(serializer=cls.__name__):
                _, messages = deprecations_while(lambda cls=cls: cls.InputParams(sample_rate=16000))

                self.assertTrue([m for m in messages if ".InputParams.sample_rate`" in m])

    def test_resampler_clear_after_secs_warns(self):
        for cls in (FrameSerializer, TwilioFrameSerializer):
            with self.subTest(params=cls.__name__):
                _, messages = deprecations_while(
                    lambda cls=cls: cls.InputParams(resampler_clear_after_secs=0.5)
                )

                self.assertTrue([m for m in messages if "resampler_clear_after_secs" in m])


class TestTwilioWireAudio(unittest.IsolatedAsyncioTestCase):
    """Twilio stands in for the μ-law serializers: they encode and decode only."""

    async def asyncSetUp(self):
        self.serializer = TwilioFrameSerializer(
            "stream-sid", params=TwilioFrameSerializer.InputParams(auto_hang_up=False)
        )

    async def test_audio_at_the_wire_rate_is_encoded(self):
        frame = OutputAudioRawFrame(audio=b"\x00\x10" * 160, sample_rate=8000, num_channels=1)

        message = json.loads(await self.serializer.serialize(frame))

        # One μ-law byte per sample, none added or lost.
        self.assertEqual(len(base64.b64decode(message["media"]["payload"])), 160)

    async def test_audio_at_another_rate_is_dropped(self):
        """Only audio serialized outside a transport arrives at another rate."""
        frame = OutputAudioRawFrame(audio=b"\x00\x10" * 480, sample_rate=24000, num_channels=1)

        self.assertIsNone(await self.serializer.serialize(frame))

    async def test_incoming_audio_keeps_the_wire_rate(self):
        """The input transport resamples it to the pipeline's rate."""
        payload = base64.b64encode(b"\xff" * 160).decode()
        message = json.dumps({"event": "media", "media": {"payload": payload}})

        frame = await self.serializer.deserialize(message)

        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.sample_rate, 8000)
        self.assertEqual(len(frame.audio), 320)
