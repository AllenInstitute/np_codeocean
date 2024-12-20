"""ETL for the dynamic routing task."""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
from aind_data_schema.components.devices import (
    Calibration,
    DAQChannel,
    DaqChannelType,
    Disc,
    Monitor,
)
from aind_data_schema.core.rig import Rig
from aind_data_schema_models.units import FrequencyUnit, SizeUnit
from aind_metadata_mapper.dynamic_routing.neuropixels_rig import (
    NeuropixelsRigContext,
    NeuropixelsRigEtl,
)

from np_codeocean.metadata import utils

logger = logging.getLogger(__name__)


class ExtractContext(NeuropixelsRigContext):
    """Extract context for DynamicRoutingTaskRigEtl."""

    version: Optional[str]
    reward_line: Optional[tuple[int, int]]
    reward_sound_line: Optional[tuple[int, int]]
    lick_line: Optional[tuple[int, int]]
    frame_signal_line: Optional[tuple[int, int]]
    acquisition_signal_line: Optional[tuple[int, int]]
    opto_channels: Optional[dict[str, list[int]]]
    galvo_channels: Optional[tuple[int, int]]
    monitor_distance: Optional[float]
    monitor_size: Optional[tuple[int, int]]
    wheel_radius: Optional[float]
    sound_calibration_fit: Optional[tuple[float, float, float]]
    solenoid_open_time: Optional[float]


# dynamic routing task has two slashes before commit hash
SUPPORTED_VERSIONS = [
    (
        b"https://raw.githubusercontent.com/samgale/DynamicRoutingTask"
        b"//9ea009a6c787c0049648ab9a93eb8d9df46d3f7b/DynamicRouting1.py"
    ),
]


class DynamicRoutingTaskRigEtl(NeuropixelsRigEtl):
    """DynamicRouting rig ETL class. Extracts information from task output
    file.
    """

    def __init__(
        self,
        input_source: Path,
        output_directory: Path,
        task_source: Path,
        monitor_name: str = "Stim",
        speaker_name: str = "Speaker",
        behavior_daq_name: str = "Behavior",
        behavior_sync_daq_name: str = "BehaviorSync",
        opto_daq_name: str = "Opto",
        reward_delivery_name: str = "Reward delivery",
        sound_calibration_date: Optional[date] = None,
        reward_calibration_date: Optional[date] = None,
        **kwargs,
    ) -> None:
        """Class constructor for Dynamic Routing rig etl class."""
        super().__init__(input_source, output_directory, **kwargs)
        self.task_source = task_source
        self.monitor_name = monitor_name
        self.speaker_name = speaker_name
        self.behavior_daq_name = behavior_daq_name
        self.behavior_sync_daq_name = behavior_sync_daq_name
        self.opto_daq_name = opto_daq_name
        self.reward_delivery_name = reward_delivery_name
        self.sound_calibration_date = sound_calibration_date
        self.reward_calibration_date = reward_calibration_date

    def _extract(self) -> ExtractContext:
        """Extracts DynamicRouting-related task information from task
        output."""
        task = utils.load_hdf5(self.task_source)
        return ExtractContext(
            current=super()._extract(),
            version=utils.extract_hdf5_value(task, ["githubTaskScript"]),
            reward_line=utils.extract_hdf5_value(task, ["rewardLine"]),
            reward_sound_line=utils.extract_hdf5_value(task, ["rewardSoundLine"]),
            lick_line=utils.extract_hdf5_value(task, ["lickLine"]),
            frame_signal_line=utils.extract_hdf5_value(task, ["frameSignalLine"]),
            acquisition_signal_line=utils.extract_hdf5_value(
                task, ["acquisitionSignalLine"]
            ),
            opto_channels=utils.extract_hdf5_value(task, ["optoChannels"]),
            galvo_channels=utils.extract_hdf5_value(task, ["galvoChannels"]),
            monitor_distance=utils.extract_hdf5_value(task, ["monDistance"]),
            monitor_size=utils.extract_hdf5_value(task, ["monSizePix"]),
            wheel_radius=utils.extract_hdf5_value(task, ["wheelRadius"]),
            sound_calibration_fit=utils.extract_hdf5_value(
                task, ["soundCalibrationFit"]
            ),
            solenoid_open_time=utils.extract_hdf5_value(task, ["solenoidOpenTime"]),
        )

    def _transform_daq_channels(
        self,
        extracted_source: ExtractContext,
        daq_name: str,
        channels: list[DAQChannel],
    ) -> None:
        """Updates channel settings for a given daq."""
        for daq in extracted_source.current.daqs:
            if daq.name == daq_name:
                logger.debug("Updating daq=%s." % daq_name)
                for channel in channels:
                    utils.find_replace_or_append(
                        daq.channels,
                        [("channel_name", channel.channel_name)],
                        channel,
                    )

    def _transform_behavior_daq(self, extracted_source: ExtractContext) -> None:
        """Updates rig model with DynamicRouting-related behavior daq
        information."""
        behavior_daq_channels = []
        if extracted_source.reward_line is not None:
            logger.debug(
                "Extracted reward line port, channel: %s" % self.behavior_daq_name
            )
            behavior_daq_channels.append(
                DAQChannel(
                    device_name=self.behavior_daq_name,
                    channel_name="solenoid",
                    channel_type=DaqChannelType.DO,
                    port=extracted_source.reward_line[0],
                    channel_index=extracted_source.reward_line[1],
                )
            )

        if extracted_source.reward_sound_line is not None:
            logger.debug(
                "Extracted reward sound line port, channel: %s, %s"
                % extracted_source.reward_sound_line
            )
            behavior_daq_channels.append(
                DAQChannel(
                    device_name=self.behavior_daq_name,
                    channel_name="reward_sound",
                    channel_type=DaqChannelType.DO,
                    port=extracted_source.reward_sound_line[0],
                    channel_index=extracted_source.reward_sound_line[1],
                )
            )

        if extracted_source.lick_line is not None:
            logger.debug(
                "Extracted lick line on port, channel: %s, %s"
                % extracted_source.lick_line
            )
            behavior_daq_channels.append(
                DAQChannel(
                    device_name=self.behavior_daq_name,
                    channel_name="lick",
                    channel_type=DaqChannelType.DI,
                    port=extracted_source.lick_line[0],
                    channel_index=extracted_source.lick_line[1],
                )
            )

        if behavior_daq_channels:
            self._transform_daq_channels(
                extracted_source, self.behavior_daq_name, behavior_daq_channels
            )

    def _transform_behavior_sync_daq(self, extracted_source: ExtractContext) -> None:
        """Updates rig model with DynamicRouting-related behavior sync daq
        information."""
        behavior_sync_daq_channels = []
        if extracted_source.frame_signal_line is not None:
            logger.debug(
                "Extracted frame signal port, channel: %s, %s"
                % extracted_source.frame_signal_line
            )
            behavior_sync_daq_channels.append(
                DAQChannel(
                    device_name=self.behavior_sync_daq_name,
                    channel_name="stim_frame",
                    channel_type=DaqChannelType.DO,
                    port=extracted_source.frame_signal_line[0],
                    channel_index=extracted_source.frame_signal_line[1],
                )
            )

        if extracted_source.acquisition_signal_line is not None:
            logger.debug(
                "Extracted aquisition port, channel: %s, %s"
                % extracted_source.acquisition_signal_line
            )
            behavior_sync_daq_channels.append(
                DAQChannel(
                    device_name=self.behavior_sync_daq_name,
                    channel_name="stim_running",
                    channel_type=DaqChannelType.DO,
                    port=extracted_source.acquisition_signal_line[0],
                    channel_index=extracted_source.acquisition_signal_line[1],
                )
            )

        if behavior_sync_daq_channels:
            self._transform_daq_channels(
                extracted_source,
                self.behavior_sync_daq_name,
                behavior_sync_daq_channels,
            )

    def _transform_opto_daq(self, extracted_source: ExtractContext) -> None:
        """Updates rig model with DynamicRouting-related opto daq
        information."""
        opto_daq_channels = []
        if extracted_source.opto_channels is not None:
            logger.debug("Extracted opto channels: %s" % extracted_source.opto_channels)
            if extracted_source.opto_channels:
                for device_name in sorted(extracted_source.opto_channels):
                    for idx, channel in enumerate(
                        extracted_source.opto_channels[device_name]
                    ):
                        opto_daq_channels.append(
                            DAQChannel(
                                device_name=self.opto_daq_name,
                                channel_name=(
                                    f"{self.opto_daq_name} " f"{device_name} #{idx}"
                                ),
                                channel_type=DaqChannelType.AO,
                                channel_index=channel,
                            )
                        )

        if extracted_source.galvo_channels is not None:
            logger.debug(
                "Extracted galvo channels x,y: %s, %s" % extracted_source.galvo_channels
            )
            opto_daq_channels.extend(
                [
                    DAQChannel(
                        device_name=self.opto_daq_name,
                        channel_name=f"{self.opto_daq_name} galvo x",
                        channel_type=DaqChannelType.AO,
                        channel_index=extracted_source.galvo_channels[0],
                        sample_rate=2000,
                        sample_rate_unit=FrequencyUnit.HZ,
                    ),
                    DAQChannel(
                        device_name=self.opto_daq_name,
                        channel_name=f"{self.opto_daq_name} galvo y",
                        channel_type=DaqChannelType.AO,
                        channel_index=extracted_source.galvo_channels[1],
                        sample_rate=2000,
                        sample_rate_unit=FrequencyUnit.HZ,
                    ),
                ]
            )

        if opto_daq_channels:
            self._transform_daq_channels(
                extracted_source, self.opto_daq_name, opto_daq_channels
            )

    def _transform_calibrations(self, extracted_source: ExtractContext) -> None:
        """Updates rig model with DynamicRouting-related calibration
        information."""
        default_calibration_date = datetime.now()

        # sound
        if extracted_source.sound_calibration_fit is not None:
            logger.debug("Updating sound calibration")
            utils.find_replace_or_append(
                extracted_source.current.calibrations,
                [
                    ("device_name", self.speaker_name),
                ],
                Calibration(
                    calibration_date=self.sound_calibration_date
                    or default_calibration_date,
                    device_name=self.speaker_name,
                    input={
                        "a": extracted_source.sound_calibration_fit[0],
                        "b": extracted_source.sound_calibration_fit[1],
                        "c": extracted_source.sound_calibration_fit[2],
                    },
                    output={},
                    description=(
                        "sound_volume = log(1 - ((dB - c) / a)) / b;"
                        "dB is sound pressure"
                    ),
                    notes=("Calibration date is a placeholder. "),
                ),
            )

        # water
        if extracted_source.solenoid_open_time is not None:
            logger.debug("Updating reward delivery calibration")
            utils.find_replace_or_append(
                extracted_source.current.calibrations,
                [
                    ("device_name", self.reward_delivery_name),
                ],
                Calibration(
                    calibration_date=self.reward_calibration_date
                    or default_calibration_date,
                    device_name=self.reward_delivery_name,
                    input={},
                    output={
                        "solenoid_open_time": (extracted_source.solenoid_open_time),
                    },
                    description=(
                        "solenoid open time (ms) = slope * expected water "
                        "volume (mL) + intercept"
                    ),
                    notes=("Calibration date is a placeholder."),
                ),
            )

    def _transform(self, extracted_source: ExtractContext) -> Rig:
        """Updates rig model with DynamicRouting-related task information."""
        if extracted_source.version is not None:
            if extracted_source.version not in SUPPORTED_VERSIONS:
                logger.warning(
                    f"Unsupported task version: {extracted_source.version}",
                )

        # monitor information
        if (
            extracted_source.monitor_distance is not None
            or extracted_source.monitor_size is not None
        ):
            for idx, device in enumerate(extracted_source.current.stimulus_devices):
                if device.name == self.monitor_name and device.device_type == "Monitor":
                    if extracted_source.monitor_distance is not None:
                        device.viewing_distance = float(
                            extracted_source.monitor_distance
                        )
                        device.viewing_distance_unit = SizeUnit.CM

                    if extracted_source.monitor_size is not None:
                        width, height = extracted_source.monitor_size
                        if not np.isnan(width) and not np.isnan(height):
                            device.width = int(width)
                            device.height = int(height)
                            device.size_unit = SizeUnit.PX

                    extracted_source.current.stimulus_devices[idx] = (
                        Monitor.model_validate(device.__dict__)
                    )
                    break

        # wheel info
        if extracted_source.wheel_radius is not None:
            logger.debug("Updating wheel information")
            extracted_source.current.mouse_platform.radius = (
                extracted_source.wheel_radius
            )
            extracted_source.current.mouse_platform.radius_unit = SizeUnit.CM
            extracted_source.current.mouse_platform = Disc.model_validate(
                extracted_source.current.mouse_platform.__dict__
            )
        # daqs
        self._transform_behavior_daq(extracted_source)
        self._transform_behavior_sync_daq(extracted_source)
        self._transform_opto_daq(extracted_source)
        # calibrations
        self._transform_calibrations(extracted_source)

        return super()._transform(extracted_source.current)