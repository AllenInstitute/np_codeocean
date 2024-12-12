import datetime
import logging

from aind_data_schema.components import devices
from aind_data_schema.core import rig
from aind_data_schema_models import organizations

from np_aind_metadata import common, rigs

logger = logging.getLogger(__name__)


def init(
    rig_name: str,
    modification_date: datetime.date,
) -> rig.Rig:
    """Initializes a rig model for an nsb behavior box.

    >>> rig_model = init("B6", datetime.date.today())

    Notes
    -----
    - rig_id is expected to be in the format:
        <ROOM NAME>_<RIG NAME>_<MODIFICATION DATE>
    - The DR task does not set the brightness and contrast of the monitor.
     These are hardcoded and assumed to be static.
    """
    if rigs.is_retrofitted_rig(rig_name):
        solenoid_valve = devices.Device(
            name="Solenoid Valve",
            device_type="Solenoid Valve",
            manufacturer=organizations.Organization.NRESEARCH_INC,
            model="161K011",
            notes="Model number is product number.",
        )
    else:
        solenoid_valve = devices.Device(
            name="Solenoid Valve",
            device_type="Solenoid Valve",
            manufacturer=organizations.Organization.NRESEARCH_INC,
            model="LVM10R1-6A-Q",
            notes="Manufacturer is a placeholder. Actual manufacturer is SMC.",
        )
    stimulus_devices = [
        devices.Monitor(
            name="Stim",
            model="PA248",
            manufacturer=organizations.Organization.ASUS,
            width=1920,
            height=1200,
            size_unit="pixel",
            viewing_distance=15.3,
            viewing_distance_unit="centimeter",
            refresh_rate=60,
            brightness=43,
            contrast=50,
        ),
        devices.RewardDelivery(
            reward_spouts=[
                devices.RewardSpout(
                    name="Reward Spout",
                    manufacturer=organizations.Organization.HAMILTON,
                    model="8649-01 Custom",
                    spout_diameter=0.672,
                    spout_diameter_unit="millimeter",
                    side=devices.SpoutSide.CENTER,
                    solenoid_valve=solenoid_valve,
                    lick_sensor=devices.Device(
                        name="Lick Sensor",
                        device_type="Lick Sensor",
                        manufacturer=organizations.Organization.OTHER,
                    ),
                    lick_sensor_type=devices.LickSensorType.PIEZOELECTIC,
                    notes=(
                        "Spout diameter is for inner diameter. "
                        "Outer diameter is 1.575mm. "
                    ),
                ),
            ]
        ),
    ]
    if rigs.is_retrofitted_rig(rig_name):
        logger.debug("Adding speaker to rig model.")
        stimulus_devices.append(
            devices.Speaker(
                name="Speaker",
                manufacturer=organizations.Organization.ISL,
                model="SPK-I-81345",
            )
        )

    room_name = rigs.get_rig_room(rig_name)
    modification_date_str = modification_date.strftime(common.MODIFICATION_DATE_FORMAT)
    rig_id = f"{room_name}_{rig_name}_{modification_date_str}"

    model = rig.Rig(
        rig_id=rig_id,
        modification_date=modification_date,
        modalities=[
            rig.Modality.BEHAVIOR,
        ],
        mouse_platform=devices.Disc(
            name="Mouse Platform",
            radius="4.69",
            radius_unit="centimeter",
            notes=(
                "Radius is the distance from the center of the wheel to the " "mouse."
            ),
        ),
        stimulus_devices=stimulus_devices,
        cameras=[],
        calibrations=[],
    )

    return rig.Rig.model_validate(model)
