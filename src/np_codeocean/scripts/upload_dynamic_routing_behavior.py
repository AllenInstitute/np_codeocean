from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime
import logging
import logging.config
import logging.handlers
import multiprocessing.synchronize
import pathlib
import multiprocessing
import multiprocessing.managers
import threading
import time
import warnings
from pathlib import Path

import h5py
import tqdm
import np_codeocean
import np_codeocean.utils
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

# Disable divide by zero or NaN warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def reset_log_file() -> None:
    log = get_log_file()
    log.parent.mkdir(exist_ok=True)
    with contextlib.suppress(OSError):
        log.unlink(missing_ok=True)
    
def get_log_file() -> pathlib.Path:
    folder = pathlib.Path(".").resolve() / "logs"
    return folder / f"{pathlib.Path(__file__).stem}_{datetime.datetime.now().strftime('%Y-%m-%d')}.log"

logging.basicConfig(
    filename=get_log_file().as_posix(),
    level=logging.INFO, 
    format="%(asctime)s | %(name)s | %(levelname)s | PID: %(process)d | TID: %(thread)d | %(message)s", 
    datefmt="%Y-%d-%m %H:%M:%S",
    )
logger = logging.getLogger(__name__)

RIG_ROOM_MAPPING = np_config.fetch('/rigs/room_numbers')
HDF5_REPO = pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting/DynamicRoutingTask/Data')
SESSION_FOLDER_DIRS = (
    pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot'),
    pathlib.Path('//allen/programs/mindscope/workgroups/templeton/TTOC/pilot recordings'),
)

EXCLUDED_SUBJECT_IDS = (0, 366122, 555555, 000000, 598796, 603810, 599657)
TASK_HDF5_GLOB = "DynamicRouting1*.hdf5"
RIG_IGNORE_PREFIXES = ("NP", "OG")

DEFAULT_HPC_UPLOAD_JOB_EMAIL = "ben.hardcastle@alleninstitute.org"

DEFAULT_DELAY_BETWEEN_UPLOADS = 40


class SessionNotUploadedError(ValueError):
    pass

class UploadLimitReachedError(RuntimeError):
    pass

def reformat_rig_model_rig_id(rig_id: str, modification_date: datetime.date) -> str:
    rig_record = npc_session.RigRecord(rig_id)
    if not rig_record.is_behavior_cluster_rig:
        raise ValueError(
            f"Only behavior boxes are supported: {rig_id=}")
    room_number = RIG_ROOM_MAPPING.get(rig_record.behavior_cluster_id, "UNKNOWN")
    return rig_record.as_aind_data_schema_rig_id(str(room_number), modification_date)


def extract_modification_date(rig: Rig) -> datetime.date:
    _, _, date_str = rig.rig_id.split("_")
    if len(date_str) == 6:
        return datetime.datetime.strptime(date_str, "%y%m%d").date()
    elif len(date_str) == 8:
        return datetime.datetime.strptime(date_str, "%Y%m%d").date()
    else:
        raise ValueError(f"Unsupported date format: {date_str}")

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
        raise FileNotFoundError("Failed to copy task rig.")
    
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
    delay: int = DEFAULT_DELAY_BETWEEN_UPLOADS,
    lock: threading.Lock | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    """
    Notes
    -----
    - task_source Path is expected to have the following naming convention:
        //allen/programs/mindscope/workgroups/dynamicrouting/DynamicRoutingTask/Data/<SUBJECT_ID>/<SESSION_ID>.hdf5
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    
    if stop_event and stop_event.is_set():
        logger.debug("Stopping due to stop event")
        return

    extracted_subject_id = npc_session.extract_subject(task_source.stem)
    if extracted_subject_id is None:
        raise SessionNotUploadedError(f"Failed to extract subject ID from {task_source}")
    logger.debug(f"Extracted subject id: {extracted_subject_id}")
    # we don't want to upload files from folders that don't correspond to labtracks IDs, like `sound`, or `*_test`
    if not task_source.parent.name.isdigit():
        raise SessionNotUploadedError(
            f"{task_source.parent.name=} is not a labtracks MID"
        )
    
    if extracted_subject_id in EXCLUDED_SUBJECT_IDS:
        raise SessionNotUploadedError(
            f"{extracted_subject_id=} is in {EXCLUDED_SUBJECT_IDS=}"
        )

    upload_root = np_session.NPEXP_ROOT / ("codeocean-dev" if test else "codeocean")
    session_dir = upload_root / f"{extracted_subject_id}_{npc_session.extract_isoformat_date(task_source.stem)}"

    np_codeocean.utils.set_npc_lims_credentials()
    try:
        session_info = npc_lims.get_session_info(task_source.stem)
    except NoSessionInfo:
        raise SessionNotUploadedError(f"{task_source.name} not in Sam's spreadsheets (yet) - cannot deduce project etc.") from None
    
    # if session has been already been uploaded, skip it
    if not (force_cloud_sync or test) and session_info.is_uploaded:  # note: session_info.is_uploaded doesnt work for uploads to dev service
        raise SessionNotUploadedError(
            f" {task_source.name} is already uploaded. Use --force-cloud-sync to re-upload."
        )
    
    # in the transfer-service airflow dag, jobs have failed after creating a folder
    # on S3, but before a data asset is created in codeocean (likely due to codeocean
    # being down): 
    # in that case, our `is_uploaded` check would return False, but in airflow,
    # there's a `check_s3_folder_exists` task, which will fail since the folder
    # already exists.
    # To avoid this second failure, we can force a re-upload, regardless of
    # whether the folder exists on S3 or not
    force_cloud_sync = True 
        
    rig_name = ""
    rig_name = session_info.training_info.get("rig_name", "")
    if not rig_name:
        with h5py.File(task_source, 'r') as file, contextlib.suppress(KeyError):
            rig_name = file['rigName'][()].decode('utf-8')
            
    if any(rig_name.startswith(i) for i in RIG_IGNORE_PREFIXES):
        raise SessionNotUploadedError(
            f"Not uploading {task_source} because rig_id starts with one of {RIG_IGNORE_PREFIXES!r}"
        )
        
    if stop_event and stop_event.is_set():
        logger.debug("Stopping due to stop event")
        return
    
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
    
    if stop_event and stop_event.is_set():
        logger.debug("Stopping due to stop event")
        return
    
    if lock is not None:
        with lock:
            if stop_event and stop_event.is_set():
                logger.debug("Stopping due to stop event")
                return
            if delay > 0:
                logger.info(f"Pausing {delay} seconds before creating upload request")
                time.sleep(delay)

    logger.info(f"Submitting {session_dir.name} to {upload_service_url}")
    
    np_codeocean.utils.put_jobs_for_hpc_upload(
        upload_jobs=np_codeocean.utils.get_job_models_from_csv(
            upload_job_path,
            user_email=hpc_upload_job_email,
        ),
        upload_service_url=upload_service_url,
        user_email=hpc_upload_job_email,
        dry_run=dry_run,
        save_path=upload_job_path.with_suffix('.json'),
    )
            
def upload_batch(
    batch_dir: pathlib.Path,
    test: bool = False,
    force_cloud_sync: bool = False,
    debug: bool = False,
    dry_run: bool = False,
    hpc_upload_job_email: str = DEFAULT_HPC_UPLOAD_JOB_EMAIL,
    delay: int = DEFAULT_DELAY_BETWEEN_UPLOADS,
    chronological_order: bool = False,
    batch_limit: int | None = None, # number of sessions to process, not upload
) -> None:
    if test:
        batch_limit = 3

    logger.addHandler(qh := logging.handlers.QueueHandler(queue := multiprocessing.Queue()))
    listener = logging.handlers.QueueListener(queue, qh)
    listener.start()
    sorted_files = tuple(
        sorted(
            batch_dir.rglob(TASK_HDF5_GLOB), 
            key=lambda p: npc_session.extract_isoformat_date(p.name), # type: ignore[return-value]
            reverse=not chronological_order,
        )
    ) # to fix tqdm we need the length of files: len(futures_dict) doesn't work for some reason
    upload_count = 0
    batch_count = 0
    future_to_task_source: dict[concurrent.futures.Future, pathlib.Path] = {}
    with (
        multiprocessing.Manager() as manager, 
        concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor,
    ):  
        sessions_remaining = manager.Value('i', batch_limit or -1)
        """Counts down and stops at zero. Set to -1 for no limit"""
        lock = manager.Lock()
        stop_event = manager.Event()
        for task_source in sorted_files:
            future = executor.submit(
                upload, 
                task_source=task_source, 
                test=test, 
                force_cloud_sync=force_cloud_sync, 
                debug=debug, 
                dry_run=dry_run, 
                hpc_upload_job_email=hpc_upload_job_email, 
                delay=delay, 
                lock=lock,
                stop_event=stop_event,
                )
            future_to_task_source[future] = task_source 
        with tqdm.tqdm(total=len(sorted_files), desc="Checking status and uploading new sessions") as pbar: 
            for future in concurrent.futures.as_completed(future_to_task_source):
                try:
                    _ = future.result()
                except SessionNotUploadedError as exc: # any other errors will be raised: prefer to fail fast when we have 12k files to process
                    logger.debug('Skipping upload of %s due to %r' % (future_to_task_source[future], exc))
                except Exception as e:
                    pbar.close()
                    logger.exception(e) 
                    raise e
                else:
                    upload_count += 1
                finally:
                    pbar.update(1) # as_completed will iterate out of order, so update tqdm progress manually
                    batch_count += 1
                    if batch_limit is not None and batch_count >= batch_limit:
                        pbar.close()
                        msg = f"Reached {batch_limit = }: stopping pending and ongoing tasks"
                        logger.info(msg)
                        print(msg)
                        stop_event.set()
                        executor.shutdown(wait=True, cancel_futures=True)
                        break
            pbar.close()
    msg = f"Batch upload complete: {upload_count} session(s) uploaded"
    logger.info(msg)
    print(msg)
    listener.stop()
    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-source', type=pathlib.Path, default=HDF5_REPO, help="Path to a single DynamicRouting1*.hdf5 file or a directory containing them (rglob will be used to find files in all subfolder levels)")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--force-cloud-sync', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--dry-run', action="store_true")
    parser.add_argument('--email', type=str, help=f"[optional] specify email address for hpc upload job updates. Default is {np_codeocean.utils.HPC_UPLOAD_JOB_EMAIL}")
    parser.add_argument('--delay', type=int, help=f"wait time (sec) between job submissions in batch mode, to avoid overloadig upload service. Default is {DEFAULT_DELAY_BETWEEN_UPLOADS}", default=DEFAULT_DELAY_BETWEEN_UPLOADS)
    parser.add_argument('--chronological', action="store_true", help="[batch mode only] Upload files in chronological order (oldest first) - default is newest first")
    parser.add_argument('--batch-limit', type=int, help="[batch mode only] Limit the number of files to upload in batch mode")
    return parser.parse_args()


def main() -> None:
    reset_log_file()
    args = parse_args()
    logger.info(f"Parsed args: {args!r}")
    if not args.task_source.is_dir():
        logger.info(f"Uploading in single file mode: {args.task_source}")
        upload(
            args.task_source,
            test=args.test,
            force_cloud_sync=args.force_cloud_sync,
            debug=args.debug,
            dry_run=args.dry_run,
            hpc_upload_job_email=args.email,
        )
    else:
        logger.info(f"Uploading in batch mode: {args.task_source}")
        upload_batch(
            batch_dir=args.task_source,
            test=args.test,
            force_cloud_sync=args.force_cloud_sync,
            debug=args.debug,
            dry_run=args.dry_run,
            hpc_upload_job_email=args.email,
            delay=args.delay,
            chronological_order=args.chronological,
            batch_limit=args.batch_limit,
        )


if __name__ == '__main__':
    main()
    # upload(
    #     task_source=pathlib.Path("//allen/programs/mindscope/workgroups/dynamicrouting/DynamicRoutingTask/Data/714753/DynamicRouting1_714753_20240703_114241.hdf5"),
    #     test=True,
    # )
    # upload(
    #     task_source=Path("//allen/programs/mindscope/workgroups/dynamicrouting/DynamicRoutingTask/Data/659250/DynamicRouting1_659250_20230322_151236.hdf5"),
    #     test=True,
    #     force_cloud_sync=True,
    #     debug=True,
    #     dry_run=False,
    # )
