#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame serialization interfaces for Pipecat."""

from abc import abstractmethod
from typing import Any

from loguru import logger
from pydantic import BaseModel, model_validator

from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.utils.base_object import BaseObject
from pipecat.utils.deprecation import warn_deprecated


class FrameSerializer(BaseObject):
    """Abstract base class for frame serialization implementations.

    Defines the interface for converting frames to/from serialized formats
    for transmission or storage. Subclasses must implement the core
    serialize/deserialize methods.

    A serializer doesn't resample audio. One whose wire format has a fixed
    sample rate states it in ``audio_in_sample_rate`` and
    ``audio_out_sample_rate``: the output transport runs at the output rate,
    so audio is resampled before it is paced, and the input transport
    resamples the audio it receives to the pipeline's rate. Audio that reaches
    the transport no faster than real time, such as live audio relayed through
    the pipeline, can still leave unevenly, since the resampler releases its
    output in blocks; producing it at the wire rate avoids resampling it.
    """

    class InputParams(BaseModel):
        """Base configuration parameters for FrameSerializer.

        Parameters:
            ignore_rtvi_messages: Whether to ignore RTVI protocol messages during serialization.
                Defaults to True to prevent RTVI messages from being sent to external transports.
            audio_in_sample_rate: Sample rate of the audio received on the wire, or None
                when the wire format carries the rate with the audio.
            audio_out_sample_rate: Sample rate of the audio sent on the wire, or None when
                the wire format carries the rate with the audio. The output transport sends
                audio at this rate.
            resampler_clear_after_secs: Has no effect: serializers don't resample.

                .. deprecated:: 1.16.0
                    No replacement. Will be removed in 2.0.0.
        """

        ignore_rtvi_messages: bool = True
        audio_in_sample_rate: int | None = None
        audio_out_sample_rate: int | None = None
        resampler_clear_after_secs: float | None = 0.2

        @model_validator(mode="before")
        @classmethod
        def _warn_resampler_clear_after_secs(cls, data: Any) -> Any:
            if isinstance(data, dict) and "resampler_clear_after_secs" in data:
                warn_deprecated(
                    "`FrameSerializer.InputParams.resampler_clear_after_secs` is deprecated "
                    "since 1.16.0 and will be removed in 2.0.0. No replacement.",
                    stacklevel=4,
                )
            return data

    def __init__(self, params: InputParams | None = None, **kwargs):
        """Initialize the FrameSerializer.

        Args:
            params: Configuration parameters.
            **kwargs: Additional arguments passed to BaseObject (e.g., name).
        """
        super().__init__(**kwargs)
        self._params = params or FrameSerializer.InputParams()
        # Audio at the wrong rate keeps arriving at the same rate, so it's
        # reported once rather than for every frame.
        self._warned_audio_out_rate = False

    @property
    def audio_in_sample_rate(self) -> int | None:
        """Sample rate of the audio received on the wire, or None if the wire format carries it."""
        return self._params.audio_in_sample_rate

    @property
    def audio_out_sample_rate(self) -> int | None:
        """Sample rate of the audio sent on the wire, or None if the wire format carries it."""
        return self._params.audio_out_sample_rate

    def should_ignore_frame(self, frame: Frame) -> bool:
        """Check if a frame should be ignored during serialization.

        This method filters out RTVI protocol messages when ignore_rtvi_messages is enabled.
        Subclasses can override this to add additional filtering logic.

        Args:
            frame: The frame to check.

        Returns:
            True if the frame should be ignored, False otherwise.
        """
        # Imported here: the rtvi package imports the transports, and the
        # transports' params import this module.
        from pipecat.processors.frameworks.rtvi.models import MESSAGE_LABEL

        if (
            self._params.ignore_rtvi_messages
            and isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame))
            and isinstance(frame.message, dict)
            and frame.message.get("label") == MESSAGE_LABEL
        ):
            return True
        return False

    async def setup(self, setup: FrameProcessorSetup):
        """Initialize the serializer with pipeline configuration.

        Args:
            setup: Configuration object containing setup parameters.
        """
        pass

    @abstractmethod
    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Convert a frame to its serialized representation.

        Args:
            frame: The frame to serialize.

        Returns:
            Serialized frame data as string or bytes, or None when there is
            nothing to send for this frame: it isn't handled, or the serializer
            is buffering it and will emit it on a later call.
        """
        pass

    @abstractmethod
    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Convert serialized data back to a frame object.

        Args:
            data: Serialized frame data as string or bytes.

        Returns:
            Reconstructed Frame object, or None if deserialization fails.
        """
        pass

    def _has_audio_out_sample_rate(self, frame: AudioRawFrame) -> bool:
        """Whether the frame's audio is at the rate sent on the wire, warning once if not."""
        rate = self.audio_out_sample_rate
        matches = rate is None or frame.sample_rate == rate
        if not matches and not self._warned_audio_out_rate:
            self._warned_audio_out_rate = True
            logger.warning(
                f"{self}: dropping {frame.sample_rate} Hz audio, the wire takes {rate} Hz. "
                f"Audio serialized outside a transport has to be resampled to {rate} Hz first."
            )
        return matches
