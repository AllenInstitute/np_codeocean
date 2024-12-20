import datetime
import logging
import shutil
import typing
from pathlib import Path

from np_codeocean.metadata import common, np, storage

logger = logging.getLogger(__name__)


def copy_rig(
    rig_name: str,
    output_path: Path,
    record_timestamp: datetime.datetime,
    storage_directory: typing.Optional[Path] = None,
) -> Path:
    """Copies a rig from storage to `output_path`.

    >>> storage_directory = Path("examples") / "rig-directory"
    >>> copy_rig("NP3", Path("rig.json"), datetime.datetime(2024, 4, 1), storage_directory)
    PosixPath('rig.json')

    Notes
    -----
    - If `storage_directory` is not provided, will attempt to get default from
     np-config.
    """
    if storage_directory is None:
        logger.debug("Getting storage directory from np-config.")
        storage_directory = np.get_rig_storage_directory()

    rig_model_path = storage.get_item(storage_directory, record_timestamp, rig_name)

    if not rig_model_path:
        raise Exception(f"No rig model found for: {rig_name}, {record_timestamp}")

    return Path(
        shutil.copy2(
            rig_model_path,
            output_path,
        )
    )


def is_behavior_box(rig_name: str | common.RigName) -> bool:
    """
    >>> is_behavior_box("NP3")
    False
    >>> is_behavior_box("B2")
    True
    """
    return not rig_name.startswith("NP")


SUPPORTED_CLUSTERS = (
    "B",
    "F",
    "D",
    "E",
)


def is_on_supported_cluster(rig_name: str) -> bool:
    """
    >>> is_on_supported_cluster("NP3")
    False
    >>> is_on_supported_cluster("B2")
    True
    """
    return rig_name[0] in SUPPORTED_CLUSTERS


NEUROPIXELS_RIG_ROOM_MAP = {
    "NP0": "325",
    "NP1": "325",
    "NP2": "327",
    "NP3": "342",
}

BEHAVIOR_CLUSTER_ROOM_MAP = {
    "B": "342",
    "F": "346",
    "G": "346",
    "D": "347",
    "E": "347",
}


def _get_rig_room(rig_name: str, room_map: dict[str, str]) -> typing.Optional[str]:
    try:
        return room_map[rig_name]
    except KeyError:
        logger.debug("No room found for rig: %s" % rig_name)
        return None


def get_rig_room(rig_name: str) -> typing.Union[str, None]:
    if is_behavior_box(rig_name):
        return _get_rig_room(rig_name[0], BEHAVIOR_CLUSTER_ROOM_MAP)
    else:
        return _get_rig_room(rig_name, NEUROPIXELS_RIG_ROOM_MAP)


def is_retrofitted_rig(rig_name: str) -> bool:
    """Retrofitted rigs are behavior box rigs that have a speaker attached and
    have a different solenoid.
    """
    return not rig_name.startswith("G")