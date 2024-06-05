import argparse
import datetime
import logging
import pathlib
from pathlib import Path

import np_config
import np_session
import np_tools
import npc_lims
import npc_session
import npc_sessions  # this is heavy, but has the logic for hdf5 -> session.json
from aind_data_schema.core.rig import Rig
from np_aind_metadata.integrations import dynamic_routing_task
from npc_lims.exceptions import NoSessionInfo

import np_codeocean
import np_codeocean.utils

logging.basicConfig(level=logging.INFO)  # TODO: move this to package __init__.py?

logger = logging.getLogger(__name__)

CONFIG = np_config.fetch('/rigs/room_numbers')
HDF5_REPO = pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting/DynamicRoutingTask/Data')
SESSION_FOLDER_DIRS = (
    pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot'),
    pathlib.Path('//allen/programs/mindscope/workgroups/templeton/TTOC/pilot recordings'),
)

EXCLUDED_SUBJECT_IDS = ("366122", "555555", "000000", "598796", "603810", "599657")
TASK_HDF5_GLOB = "DynamicRouting1*.hdf5"
IGNORE_PREFIX = "NP"

DEFAULT_HPC_UPLOAD_JOB_EMAIL = "chrism@alleninstitute.org"

def reformat_rig_model_rig_id(rig_id: str, modification_date: datetime.date) -> str:
    rig_record = npc_session.RigRecord(rig_id)
    if not rig_record.is_behavior_cluster_rig:
        raise Exception(
            f"Rig is not a behavior cluster rig. Only behavior cluster rigs are supported. rig_id={rig_id}")
    room_number = CONFIG.get(rig_record.behavior_cluster_id, "UNKNOWN")
    return rig_record.as_aind_data_schema_rig_id(str(room_number), modification_date)


def extract_modification_date(rig: Rig) -> datetime.date:
    _, _, date_str = rig.rig_id.split("_")
    if len(date_str) == 6:
        return datetime.datetime.strptime(date_str, "%y%m%d").date()
    elif len(date_str) == 8:
        return datetime.datetime.strptime(date_str, "%Y%m%d").date()
    else:
        raise Exception(f"Unsupported date format: {date_str}")


def wrapped_get_session_info(
    task_source: pathlib.Path,
) -> npc_lims.SessionInfo | None:
    try:
        return npc_lims \
            .get_session_info(task_source.stem)
    except NoSessionInfo:
        logger.debug(
            f"Skipping {task_source} because session info not Dynamic Routing Task"
        )
    except Exception:
        logger.debug(
            f"Skipping {task_source} because of exception.",
            exc_info=True,
        )

    return None


def add_metadata(
    task_source: pathlib.Path,
    dest: pathlib.Path,
    rig_storage_directory: pathlib.Path,
):
    """Adds `aind-data-schema` rig and session metadata to a session directory.
    """
    # we need to patch due to this bug not getting addressed: https://github.com/AllenInstitute/npc_sessions/pull/103
    # npc_sessions.Session._aind_rig_id = property(aind_rig_id_patch)
    npc_sessions.Session(task_source) \
        ._aind_session_metadata.write_standard_file(dest)
    
    session_metadata_path = dest / "session.json"
    rig_metadata_path = dynamic_routing_task.copy_task_rig(
        task_source,
        dest / "rig.json",
        rig_storage_directory,
    )
    if not rig_metadata_path:
        raise Exception("Failed to copy task rig.")
    
    rig_metadata = Rig.model_validate_json(rig_metadata_path.read_text())
    modification_date = datetime.date(2024, 4, 1)  # keep cluster rigs static for now
    rig_metadata.modification_date = modification_date
    rig_metadata.rig_id = reformat_rig_model_rig_id(rig_metadata.rig_id, modification_date)
    rig_metadata.write_standard_file(dest)  # assumes this will work out to dest/rig.json
    
    dynamic_routing_task.update_session_from_rig(
        session_metadata_path,
        rig_metadata_path,
        session_metadata_path,
    )


def upload(
    task_source: Path,
    test: bool = False,
    force_cloud_sync: bool = False,
    debug: bool = False,
    dry_run: bool = False,
    hpc_upload_job_email: str = DEFAULT_HPC_UPLOAD_JOB_EMAIL,
) -> Path | None:
    """
    Notes
    -----
    - task_source Path is expected to have the following naming convention:
        //allen/programs/mindscope/workgroups/dynamicrouting/DynamicRoutingTask/Data/<SUBJECT_ID>/<SESSION_ID>.hdf5
    """
    if debug:
        logger.setLevel(logging.DEBUG)

    extracted_subject_id = npc_session.extract_subject(task_source.stem)
    logger.debug(f"Extracted subject id: {extracted_subject_id}")
    # we don't want to upload files from folders that don't correspond to labtracks IDs, like `sound`, or `*_test`
    if not extracted_subject_id.isdigit():
        logger.debug(
            f"Skipping {task_source} because parent folder name is not a number")
        return None
    
    if extracted_subject_id in EXCLUDED_SUBJECT_IDS:
        logger.debug(
            f"Skipping {task_source} because subject ID is in EXCLUDED_SUBJECT_IDS")
        return None
    
    session_info = wrapped_get_session_info(task_source)
    if not session_info:
        return None

    if session_info.training_info["rig_name"].startswith(IGNORE_PREFIX):
        logger.debug(
            f"Skipping {task_source} because rig_id starts with {IGNORE_PREFIX}")
        return None

    # if session has been already been uploaded, skip it
    if not test and session_info.is_uploaded:  # session_info.is_uploaded doesnt work for uploads to dev service
        if force_cloud_sync:
            logger.info(
                f"Session {task_source} has already been uploaded, but force_cloud_sync={force_cloud_sync}. Re-uploading.")
        else:
            logger.info(
                f"Session {task_source} has already been uploaded. Skipping.")
            return None

    upload_root = np_session.NPEXP_ROOT / "codeocean-dev" if test else "codeocean"
    session_dir = upload_root / session_info.id
    logger.debug(f"Session upload directory: {session_dir}")

    # external systems start getting modified here.
    session_dir.mkdir(exist_ok=True)
    metadata_dir = session_dir / 'aind_metadata'
    metadata_dir.mkdir(exist_ok=True)
    behavior_modality_dir = session_dir / "behavior"
    behavior_modality_dir.mkdir(exist_ok=True)

    rig_storage_directory = np_codeocean.get_project_config()["rig_metadata_dir"]
    logger.debug(f"Rig storage directory: {rig_storage_directory}")
    add_metadata(
        task_source,
        metadata_dir,
        rig_storage_directory=rig_storage_directory,
    )

    np_tools.symlink(
        np_codeocean.utils.ensure_posix(task_source),
        behavior_modality_dir / task_source.name,
    )

    upload_job_contents = {
        'subject-id': extracted_subject_id,
        'acq-datetime': npc_session.extract_isoformat_datetime(task_source.stem),
        'project_name': 'Dynamic Routing',
        'platform': 'behavior',
        'modality0': 'behavior',
        'metadata_dir': np_config.normalize_path(metadata_dir).as_posix(),
        'modality0.source': np_config.normalize_path(
            behavior_modality_dir).as_posix(),
        'force_cloud_sync': force_cloud_sync,
    }

    upload_job_path = np_codeocean.write_upload_csv(
        upload_job_contents,
        np_config.normalize_path(session_dir / 'upload.csv'),
    )

    upload_service_url = np_codeocean.utils.DEV_SERVICE \
        if test else np_codeocean.utils.AIND_DATA_TRANSFER_SERVICE
    logger.debug(f"Uploading to: {upload_service_url}")
    
    np_codeocean.utils.put_csv_for_hpc_upload(
        csv_path=upload_job_path,
        upload_service_url=upload_service_url,
        hpc_upload_job_email=hpc_upload_job_email,
        dry_run=dry_run,
    )
    return upload_job_path


def upload_batch(
    batch_dir: pathlib.Path,
    test: bool = False,
    force_cloud_sync: bool = False,
    debug: bool = False,
    dry_run: bool = False,
    hpc_upload_job_email: str = DEFAULT_HPC_UPLOAD_JOB_EMAIL,
) -> None:
    if test:
        batch_limit = 3
    else:
        batch_limit = None
    upload_count = 0
    for task_source in batch_dir.rglob(TASK_HDF5_GLOB):
        logger.info("Uploading %s" % task_source)
        upload_job_path = upload(
            task_source,
            test=test,
            force_cloud_sync=force_cloud_sync,
            debug=debug,
            dry_run=dry_run,
            hpc_upload_job_email=hpc_upload_job_email,
        )
        if batch_limit is not None and upload_job_path is not None:
            upload_count += 1
            if upload_count >= batch_limit:
                logger.info(f"Reached batch limit of {batch_limit}. Exiting.")
                break


MODES = ['singleton', 'batch']
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-source', type=pathlib.Path)
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--force-cloud-sync', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--dry-run', action="store_true")
    parser.add_argument('--mode', default=MODES[0], choices=MODES)
    parser.add_argument('--batch-dir', type=pathlib.Path, default=HDF5_REPO)
    parser.add_argument('--email', type=str, help=f"[optional] specify email address for hpc upload job updates. Default is {np_codeocean.utils.HPC_UPLOAD_JOB_EMAIL}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == MODES[0]:
        logger.info(f"Uploading in singleton mode: {args.task_source}")
        if not args.task_source:
            raise Exception("Task source is required for singleton mode.")
        upload(
            args.task_source,
            test=args.test,
            force_cloud_sync=args.force_cloud_sync,
            debug=args.debug,
            dry_run=args.dry_run,
            hpc_upload_job_email=args.email,
        )
    elif args.mode == MODES[1]:
        logger.info(f"Uploading in match mode: {args.batch_dir}")
        upload_batch(
            batch_dir=args.batch_dir,
            test=args.test,
            force_cloud_sync=args.force_cloud_sync,
            debug=args.debug,
            dry_run=args.dry_run,
            hpc_upload_job_email=args.email,
        )
    else:
        raise Exception(f"Unexpected mode: {args.mode}")


if __name__ == '__main__':
    main()
