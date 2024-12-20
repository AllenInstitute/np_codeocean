import ast
import datetime
import logging
import os
import pathlib
import typing

from aind_data_schema.core import rig

from np_codeocean.metadata import common, storage
from np_codeocean.metadata.model_templates import neuropixels_rig

logger = logging.getLogger(__name__)

try:
    import np_config
except Exception:
    logger.error("Failed to import neuropixels-related dependencies.", exc_info=True)


# cannot type hint due to np import failing in github actions
def _get_rig_config(rig_name: common.RigName):
    return np_config.Rig(ast.literal_eval(rig_name[-1]))


def get_manipulator_infos(
    rig_name: common.RigName,
) -> list[common.ManipulatorInfo]:
    return [
        common.ManipulatorInfo(
            assembly_name=f"Ephys Assembly {key}",
            serial_number=value,
        )
        for key, value in _get_rig_config(rig_name)
        .config["services"]["NewScaleCoordinateRecorder"]["probe_to_serial_number"]
        .items()
    ]


def init_neuropixels_rig_from_np_config(
    rig_name: common.RigName,
    modification_date: typing.Optional[datetime.date] = None,
) -> rig.Rig:
    """Initializes a rig model using settings from np_config.

    Notes
    -----
    - Might require you to be onprem to connect to np_config's zookeeper
     server.
    """
    rig_config = _get_rig_config(rig_name)
    return neuropixels_rig.init(
        rig_name,
        mon_computer_name=rig_config.Mon,
        stim_computer_name=rig_config.Stim,
        sync_computer_name=rig_config.Sync,
        modification_date=modification_date,
    )


def get_rig_storage_directory() -> pathlib.Path:
    """Convenience function to get the default rig storage directory from
    np-config. Only works onprem.
    """
    config = np_config.fetch("/projects/np-aind-metadata")
    storage_directory_parts = config["rig_storage_directory_parts"]
    storage_directory_root = os.getenv("STORAGE_DIRECTORY_ROOT")
    if storage_directory_root:
        logger.debug("Overriding storage directory root: %s" % storage_directory_root)
        storage_directory_parts[0] = storage_directory_root
    return pathlib.Path(*storage_directory_parts)


def get_current_rig_model(
    rig_name: common.RigName,
) -> pathlib.Path | None:
    """Gets path to current rig model.

    Notes
    -----
    - Probably only works onprem. Needs to connect to zookeeper server but also
    potentially network storage.
    """
    storage_directory = get_rig_storage_directory()
    logger.debug("Storage directory: %s" % storage_directory)
    return storage.get_item(
        storage_directory,
        datetime.datetime.now(),
        rig_name,
    )