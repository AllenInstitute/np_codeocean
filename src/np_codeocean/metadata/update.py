
import datetime
import logging
import pathlib
import shutil
import tempfile
import typing

from aind_data_schema.core import rig
from aind_metadata_mapper.dynamic_routing import (
    mvr_rig,
    neuropixels_rig,
    sync_rig,
)
from aind_metadata_mapper.open_ephys.rig import OpenEphysRigEtl

from np_codeocean.metadata import common, utils
from np_codeocean.metadata.dynamic_routing_task_etl import DynamicRoutingTaskRigEtl

logger = logging.getLogger(__name__)


def _run_neuropixels_rig_etl(
    etl: neuropixels_rig.NeuropixelsRigEtl,
) -> None:
    """Utility for running a neuropixels rig ETL and continuing on error."""
    try:
        etl.run_job()
    except Exception:
        logger.error("Error calling: %s" % etl, exc_info=True)


def update_rig_modification_date(
    current_model_path: pathlib.Path,
    modification_date: datetime.date,
) -> pathlib.Path:
    """Convenience function that updates the modification date of an
     `aind-data-schema` `rig.json` and saves it over its current path.

    >>> update_rig_modification_date(
    ...     pathlib.Path(".", "examples", "rig.json"),
    ...     datetime.date(2024, 4, 1),
    ... )
    PosixPath('examples/rig.json')
    """
    model = rig.Rig.model_validate_json(current_model_path.read_text())
    id_parts = model.rig_id.split("_")
    id_parts[-1] = modification_date.strftime(common.MODIFICATION_DATE_FORMAT)
    model.rig_id = "_".join(id_parts)
    model.modification_date = modification_date
    return utils.save_aind_model(model, current_model_path)


def update_rig_room_number(
    current_model_path: pathlib.Path,
    room_number: str,
) -> pathlib.Path:
    """Convenience function that updates the room number of an
     `aind-data-schema` `rig.json` and saves it over its current path.

    >>> update_rig_room_number(
    ...     pathlib.Path(".", "examples", "rig.json"),
    ...     "324",
    ... )
    PosixPath('examples/rig.json')
    """
    model = rig.Rig.model_validate_json(current_model_path.read_text())
    id_parts = model.rig_id.split("_")
    id_parts[0] = room_number
    model.rig_id = "_".join(id_parts)
    return utils.save_aind_model(model, current_model_path)


def update_rig(
    rig_source: pathlib.Path,
    modification_date: typing.Optional[datetime.date] = None,
    task_source: typing.Optional[pathlib.Path] = None,
    reward_calibration_date: typing.Optional[datetime.date] = None,
    sound_calibration_date: typing.Optional[datetime.date] = None,
    open_ephys_settings_sources: typing.Optional[list[pathlib.Path]] = None,
    sync_source: typing.Optional[pathlib.Path] = None,
    mvr_source: typing.Optional[pathlib.Path] = None,
    mvr_mapping: dict[str, str] = common.DEFAULT_MVR_MAPPING,
    output_path: pathlib.Path = pathlib.Path("rig.json"),
) -> pathlib.Path:
    """Updates a rig model file with the metadata from the given sources.

    >>> update_rig(
    ...  pathlib.Path(".", "examples", "rig.json"),
    ...  open_ephys_settings_sources=[
    ...     pathlib.Path(".", "examples", "settings.xml"),
    ...  ],
    ... )
    PosixPath('rig.json')

    Notes
    -----
    - *_source, if present will update various values in the rig model.
    """
    # build model in a temporary directory
    build_dir = pathlib.Path(tempfile.mkdtemp())
    initial_model = rig.Rig.model_validate_json(rig_source.read_text())
    build_source = pathlib.Path(shutil.copy2(rig_source, build_dir))
    logger.debug("Rig build source: %s" % build_source)
    if task_source:
        logger.debug("Updating rig model with dynamic routing task context.")
        _run_neuropixels_rig_etl(
            DynamicRoutingTaskRigEtl(
                build_source,
                build_dir,
                task_source=task_source,
                sound_calibration_date=sound_calibration_date,
                reward_calibration_date=reward_calibration_date,
            )
        )

    if open_ephys_settings_sources:
        logger.debug("Updating rig model with open ephys context.")
        _run_neuropixels_rig_etl(
            OpenEphysRigEtl(
                build_source,
                build_dir,
                open_ephys_settings_sources=open_ephys_settings_sources,
            )
        )

    if sync_source:
        logger.debug("Updating rig model with sync context.")
        _run_neuropixels_rig_etl(
            sync_rig.SyncRigEtl(
                build_source,
                build_dir,
                config_source=sync_source,
            )
        )

    if mvr_source and mvr_mapping:
        logger.debug("Updating rig model with mvr context.")
        _run_neuropixels_rig_etl(
            mvr_rig.MvrRigEtl(
                build_source,
                build_dir,
                mvr_config_source=mvr_source,
                mvr_mapping=mvr_mapping,
            )
        )

    updated_model = rig.Rig.model_validate_json(build_source.read_text())

    if updated_model != initial_model:
        if modification_date:
            update_rig_modification_date(build_source, modification_date)
        else:
            update_rig_modification_date(build_source, datetime.date.today())

    return pathlib.Path(shutil.copy2(build_source, output_path))