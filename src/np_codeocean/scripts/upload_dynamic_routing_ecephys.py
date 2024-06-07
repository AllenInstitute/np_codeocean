import argparse
import datetime
import logging
import typing
from pathlib import Path

import np_config
import np_session
import npc_session
import npc_sessions
from aind_data_schema.core.rig import Rig
from np_aind_metadata.integrations import dynamic_routing_task

import np_codeocean
import np_codeocean.utils

logger = logging.getLogger(__name__)

CONFIG = np_config.fetch('/rigs/room_numbers')


def reformat_rig_model_rig_id(rig_id: str, modification_date: datetime.date) -> str:
    rig_record = npc_session.RigRecord(rig_id)
    if not rig_record.is_neuro_pixels_rig:
        raise Exception(
            f"Rig is not a neuropixels rig. Only behavior cluster rigs are supported. rig_id={rig_id}")
    room_number = CONFIG.get(rig_record, "UNKNOWN")
    return rig_record.as_aind_data_schema_rig_id(str(room_number), modification_date)


def extract_modification_date(rig: Rig) -> datetime.date:
    _, _, date_str = rig.rig_id.split("_")
    if len(date_str) == 6:
        return datetime.datetime.strptime(date_str, "%y%m%d").date()
    elif len(date_str) == 8:
        return datetime.datetime.strptime(date_str, "%Y%m%d").date()
    else:
        raise Exception(f"Unsupported date format: {date_str}")


def add_metadata(
    session_directory: Path,
    session_datetime: datetime.datetime,
    platform: np_codeocean.utils.AINDPlatform,
    rig_storage_directory: Path,
) -> None:
    """Adds rig and sessions metadata to a session directory.

    TODO: Return created paths rather than None to better support a monadic
     pattern.
    """
    normalized = np_config.normalize_path(session_directory)
    logger.debug("Normalized session directory: %s" % normalized)
    
    session_json = normalized / "session.json"
    if not (session_json.is_symlink() or session_json.exists()):
        logger.debug("Attempting to create session.json")
    try:
        npc_sessions.DynamicRoutingSession(normalized)._aind_session_metadata.write_standard_file(normalized)
    except Exception as e:
            logger.exception(e)
    else:
        if session_json.exists():
            logger.debug("Created session.json")
        else:
                logger.warning("Failed to find created session.json, but no error occurred during creation: may be in unexpected location")
    if not (session_json.is_symlink() or session_json.exists()):
        logger.warning("session.json is currently required for the rig.json to be created, so we can't continue with metadata creation")
        return None

    if platform in ('ecephys', ):
        dynamic_routing_task.add_np_rig_to_session_dir(
            normalized,
            session_datetime,
            rig_storage_directory,
        )
        rig_model_path = normalized / "rig.json"
    elif platform in ('behavior', ):
        task_paths = list(
            normalized.glob("Dynamic*.hdf5")
        )
        logger.debug("Scraped task_paths: %s" % task_paths)
        task_path = task_paths[0]
        logger.debug("Using task path: %s" % task_path)
        rig_model_path = dynamic_routing_task.copy_task_rig(
            task_path,
            normalized / "rig.json",
            rig_storage_directory,
        )
        if not rig_model_path:
            raise Exception("Failed to copy task rig.")
        logger.debug("Rig model path: %s" % rig_model_path)
        session_model_path = dynamic_routing_task.scrape_session_model_path(
            normalized,
        )
        logger.debug("Session model path: %s" % session_model_path)
        dynamic_routing_task.update_session_from_rig(
            session_model_path,
            rig_model_path,
            session_model_path,
        )
    else:
        raise Exception("Unexpected platform: %s" % platform)
    
    assert rig_model_path.exists(), \
            f"Rig metadata path does not exist: {rig_model_path}"

    rig_metadata = Rig.model_validate_json(rig_model_path.read_text())
    modification_date = extract_modification_date(rig_metadata)
    rig_metadata.rig_id = reformat_rig_model_rig_id(rig_metadata.rig_id, modification_date)
    rig_metadata.write_standard_file(normalized)  # assumes this will work out to dest/rig.json
    session_model_path = dynamic_routing_task.scrape_session_model_path(
        normalized,
    )
    dynamic_routing_task.update_session_from_rig(
        session_model_path,
        rig_model_path,
        session_model_path,
    )

    return None


def write_metadata_and_upload(
    session: str | int | Path | np_session.Session, 
    recording_dirs: typing.Iterable[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    test: bool = False,
    hpc_upload_job_email: str = np_codeocean.utils.HPC_UPLOAD_JOB_EMAIL,
) -> None:
    """Writes and updates aind-data-schema to the session directory
     associated with the `session`. The aind-data-schema session model is
     updated to reflect the `rig_id` of the rig model added to the session
     directory.
    
    Only handles ecephys platform uploads (ie sessions with a folder of data; not 
    behavior box sessions, which have a single hdf5 file only)
    """
    session = np_session.Session(session)
    platform: np_codeocean.utils.AINDPlatform = 'ecephys'
    logger.debug(f"Platform: {platform}")
    rig_storage_directory = np_codeocean.get_project_config()["rig_metadata_dir"]
    logger.debug(f"Rig storage directory: {rig_storage_directory}")
    add_metadata(
        session_directory=session.npexp_path,
        session_datetime=session.start,
        platform='ecephys',
        rig_storage_directory=rig_storage_directory,
    )
    return np_codeocean.upload_session(
        session,
        recording_dirs=recording_dirs,
        force=force,
        dry_run=dry_run,
        test=test,
        hpc_upload_job_email=hpc_upload_job_email,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a session to CodeOcean")
    parser.add_argument('session', help="session ID (lims or np-exp foldername) or path to session folder")
    parser.add_argument('recording_dirs', nargs='*', type=list, help="[optional] specific recording directories to upload - for use with split recordings only.")
    parser.add_argument('--email', dest='hpc_upload_job_email', type=str, help=f"[optional] specify email address for hpc upload job updates. Default is {np_codeocean.utils.HPC_UPLOAD_JOB_EMAIL}")
    parser.add_argument('--force', action='store_true', help="enable `force_cloud_sync` option, re-uploading and re-making raw asset even if data exists on S3")
    parser.add_argument('--test', action='store_true', help="use the test-upload service, uploading to the test CodeOcean server instead of the production server")
    parser.add_argument('--dry-run', action='store_true', help="Create upload job but do not submit to hpc upload queue.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    write_metadata_and_upload(**vars(args))


if __name__ == '__main__':
    main()
