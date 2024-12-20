"""Convenience functions for:
- Adding neuropixels rig to dynamic routing session directory.
- Updating neuropixels rig from dynamic routing session directory.
"""

import datetime
import logging
import pathlib
import typing

from aind_data_schema.core import rig, session

from np_codeocean.metadata import common, np, rigs, storage, update, utils
from np_codeocean.metadata.model_templates import behavior_box, neuropixels_rig

logger = logging.getLogger(__name__)


SESSION_MODEL_GLOB_PATTERN = "*session.json"


def scrape_session_model_path(session_directory: pathlib.Path) -> pathlib.Path:
    """Scrapes aind-metadata session json from dynamic routing session
    directory.
    """
    matches = list(session_directory.glob(SESSION_MODEL_GLOB_PATTERN))
    logger.debug("Scraped session model paths: %s" % matches)
    return matches[0]


def update_session_from_rig(
    session_source: pathlib.Path,
    rig_source: pathlib.Path,
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Convenience function that updates the `rig_id` of a session model at
     `session_source`. Uses the `rig_id` of `rig_source`.

    Notes
    -----
    - Overwrites the session model at `output_path`.
    """
    session_model = session.Session.model_validate_json(
        session_source.read_text())
    rig_model = rig.Rig.model_validate_json(rig_source.read_text())
    session_model.rig_id = rig_model.rig_id
    return utils.save_aind_model(session_model, output_path)


def copy_or_init_rig(
    storage_directory: pathlib.Path,
    extracted_session_datetime: datetime.datetime,
    extracted_rig_name: str,
    output_path: pathlib.Path,
) -> pathlib.Path:
    try:
        rig_model_path = storage.get_item(
            storage_directory, extracted_session_datetime, extracted_rig_name
        )
        # validate that existing model is of the correct current
        #  aind-data-schema metadata format
        assert rig_model_path is not None
        rig.Rig.model_validate_json(rig_model_path.read_text())

        return rigs.copy_rig(
            extracted_rig_name,
            output_path,
            extracted_session_datetime,
            storage_directory,
        )
    except Exception:
        logger.error("Failed to copy rig.", exc_info=True)
        rig_model = neuropixels_rig.init(
            extracted_rig_name,  # type: ignore
            modification_date=datetime.date.today(),
        )
        rig_model.write_standard_file(output_path.parent)
        return output_path


def add_np_rig_to_session_dir(
    session_dir: pathlib.Path,
    session_datetime: datetime.datetime,
    rig_model_dir: typing.Optional[pathlib.Path] = None,
) -> None:
    """Direct support for the dynamic routing task. Adds an `aind-data-schema`
     `rig.json` to a dynamic routing session directory. The `aind-data-schema`
     `session.json` in `session_dir` will be updated with the `rig_id` of the
      added `rig.json`.

    Notes
    -----
    - An aind metadata session json must exist and be ending with filename
    session.json (pattern: `*session.json`) in `session_dir`.
    - If `rig_model_dir` is not provided, will attempt to get default from
     np-config. You will need to be onprem for `np-config` to work.
    """
    scraped_session_model_path = scrape_session_model_path(session_dir)
    logger.debug("Scraped session model path: %s" % scraped_session_model_path)
    scraped_session = session.Session.model_validate_json(
        scraped_session_model_path.read_text()
    )
    scraped_rig_id = scraped_session.rig_id
    logger.info("Scraped rig id: %s" % scraped_rig_id)
    _, rig_name, _ = scraped_rig_id.split("_")
    logger.info("Parsed rig name: %s" % rig_name)
    rig_model_path = session_dir / "rig.json"

    if not rig_model_dir:
        logger.debug("Getting storage directory from np-config.")
        rig_model_dir = np.get_rig_storage_directory()

    current_model_path = copy_or_init_rig(
        rig_model_dir,
        session_datetime,
        rig_name,
        rig_model_path,
    )

    logger.info("Current model path: %s" % current_model_path)
    settings_sources = list(session_dir.glob("**/settings.xml"))
    logger.info("Scraped open ephys settings: %s" % settings_sources)

    updated_model_path = update.update_rig(
        rig_model_path,
        modification_date=session_datetime.date(),
        open_ephys_settings_sources=settings_sources,
        output_path=rig_model_path,
    )

    update_session_from_rig(
        scraped_session_model_path,
        updated_model_path,
        scraped_session_model_path,
    )

    storage.update_item(
        rig_model_dir,
        updated_model_path,
        session_datetime,
        rig_name,
    )


def update_neuropixels_rig_from_dynamic_routing_session_dir(
    rig_source: pathlib.Path,
    session_dir: pathlib.Path,
    output_path: pathlib.Path = pathlib.Path("rig.json"),
    modification_date: typing.Optional[datetime.date] = None,
    mvr_mapping: dict[str, str] = common.DEFAULT_MVR_MAPPING,
) -> pathlib.Path:
    """Scrapes dynamic routing session directory for various rig
    configuration/settings and updates `rig_source`.

    Notes
    -----
    - Will likely be depreciated in the future.
    """
    try:
        task_source = next(session_dir.glob("**/Dynamic*.hdf5"))
        logger.debug("Scraped task source: %s" % task_source)
    except StopIteration:
        task_source = None

    # sync
    try:
        sync_source = next(session_dir.glob("**/sync.yml"))
        logger.debug("Scraped sync source: %s" % sync_source)
    except StopIteration:
        sync_source = None

    # mvr
    try:
        mvr_source = next(session_dir.glob("**/mvr.ini"))
        logger.debug("Scraped mvr source: %s" % mvr_source)
    except StopIteration:
        mvr_source = None

    # open ephys
    settings_sources = list(session_dir.glob("**/settings.xml"))
    logger.debug("Scraped open ephys settings: %s" % settings_sources)

    return update.update_rig(
        rig_source,
        task_source=task_source,
        sync_source=sync_source,
        mvr_source=mvr_source,
        mvr_mapping=mvr_mapping,
        open_ephys_settings_sources=settings_sources,
        output_path=output_path,
        modification_date=modification_date,
        reward_calibration_date=modification_date,
        sound_calibration_date=modification_date,
    )


def extract_rig_name(task_source: pathlib.Path) -> str | None:
    """Extracts rig_name from task_source.

    >>> extract_rig_name(
    ...     pathlib.Path("examples") / "neuropixels-rig-task.hdf5"
    ... )
    'NP2'
    >>> extract_rig_name(
    ...     pathlib.Path("examples") / "behavior-box-task-0.hdf5"
    ... )
    'D6'
    >>> extract_rig_name(
    ...     pathlib.Path("examples") / "behavior-box-task-1.hdf5"
    ... )
    'B2'

    Notes
    -----
    - If extracted `computerName` is not found or is not bytes, will use
     `rigName`.
    """
    computer_name = utils.extract_hdf5_value(
        utils.load_hdf5(task_source),
        [
            "computerName",
        ],
    )
    logger.debug("Extracted computerName: %s" % computer_name)
    rig_name = utils.extract_hdf5_value(
        utils.load_hdf5(task_source),
        [
            "rigName",
        ],
    )
    logger.debug("Extracted rigName: %s" % rig_name)

    if isinstance(computer_name, bytes):
        decoded = computer_name.decode("utf8")
        if decoded.lower().startswith("beh"):
            postfixed = decoded.split(".")[1]
            split = postfixed.split("-")
            return split[0] + split[1][-1]
        else:
            return decoded

    if isinstance(rig_name, bytes):
        return rig_name.decode("utf-8")

    return None


def extract_session_datetime(
    task_source: pathlib.Path,
) -> datetime.datetime:
    """
    >>> extract_session_datetime(
    ...     pathlib.Path("examples") / "behavior-box-task-0.hdf5"
    ... )
    datetime.datetime(2024, 5, 1, 0, 0)
    >>> extract_session_datetime(
    ...     pathlib.Path("examples") / "behavior-box-task-1.hdf5"
    ... )
    datetime.datetime(2023, 9, 8, 0, 0)
    """
    start_time_str = utils.extract_hdf5_value(
        utils.load_hdf5(task_source),
        [
            "startTime",
        ],
    )
    if not start_time_str:
        raise Exception("Could not extract start time from task source.")

    logger.debug("Extracted start time bytes: %s" % start_time_str)
    date_str, _ = start_time_str.decode("utf8").split("_")
    logger.debug("Date string: %s" % date_str)
    return datetime.datetime.strptime(date_str, "%Y%m%d")


def copy_task_rig(
    task_source: pathlib.Path,
    output_path: pathlib.Path,
    storage_directory: typing.Optional[pathlib.Path] = None,
) -> pathlib.Path | None:
    """Extracts rig_name from task_source and copies the associated `rig.json`
     to output_path.

    >>> storage_directory = pathlib.Path("examples") / "rig-directory"
    >>> task_source = pathlib.Path("examples") / "neuropixels-rig-task.hdf5"
    >>> copy_task_rig(
    ...     task_source,
    ...     pathlib.Path("rig.json"),
    ...     storage_directory,
    ... )
    PosixPath('rig.json')

    >>> task_source = pathlib.Path("examples") / "behavior-box-task-0.hdf5"
    >>> copy_task_rig(
    ...     task_source,
    ...     pathlib.Path("rig.json"),
    ...     storage_directory,
    ... )
    PosixPath('rig.json')

    Notes
    -----
    - If `storage_directory` is not provided, will attempt to get default from
     np-config.
    """
    # storage_directory optional is legacy behavior
    # TODO: remove the optional so we can remove this safeguard
    if not storage_directory:
        raise Exception("Storage directory must be provided.")

    extracted_rig_name = extract_rig_name(task_source)
    logger.debug("Extracted rig name: %s" % extracted_rig_name)
    if not extracted_rig_name:
        raise Exception("Could not extract rig name from task source: %s" % task_source)

    if rigs.is_behavior_box(extracted_rig_name):
        rig_model = behavior_box.init(
            extracted_rig_name,
            modification_date=datetime.date.today(),
        )
        rig_model.write_standard_file(output_path.parent)
        return output_path

    extracted_session_datetime = extract_session_datetime(task_source)

    # if grabbing latest rig model fails, return a new one
    return copy_or_init_rig(
        storage_directory,
        extracted_session_datetime,
        extracted_rig_name,
        output_path,
    )
