import csv
import argparse
import logging
import pathlib
import typing
from pathlib import Path

import np_config
import np_session
import np_tools
import npc_lims
import npc_sessions  # this is heavy, but has the logic for hdf5 -> session.json

from np_aind_metadata.integrations import dynamic_routing_task

from np_codeocean import upload as np_codeocean_upload


logger = logging.getLogger(__name__)


HDF5_REPO = pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting/DynamicRoutingTask/Data')
SESSION_FOLDER_DIRS = (
    pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot'),
    pathlib.Path('//allen/programs/mindscope/workgroups/templeton/TTOC/pilot recordings'),
)

EXCLUDED_SUBJECT_IDS = ("366122", "555555", "000000", "598796", "603810", "599657")
TASK_HDF5_GLOB = "DynamicRouting1*.hdf5"
ACQ_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def write_upload_job(
    content: dict[str, typing.Any],
    output_path: pathlib.Path,
) -> pathlib.Path:
    with open(output_path, 'w') as f:
        w = csv.writer(f, lineterminator='')
        w.writerow(content.keys())
        w.writerow('\n')
        w.writerow(content.values())
    return output_path


def add_metadata(
    task_source: pathlib.Path,
    dest: pathlib.Path,
    rig_storage_directory: pathlib.Path,
):
    """Adds `aind-data-schema` rig and session metadata to a session directory.
    """
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
    
    dynamic_routing_task.update_session_from_rig(
        session_metadata_path,
        rig_metadata_path,
        session_metadata_path,
    )


def upload(
    task_source: Path,
    test: bool = False,
    force_cloud_sync: bool = False,
) -> Path | None:
    """
    Notes
    -----
    - task_source Path is expected to have the following naming convention:
        //allen/programs/mindscope/workgroups/dynamicrouting/DynamicRoutingTask/Data/<SUBJECT_ID>/<SESSION_ID>.hdf5
    """
    extracted_subject_id = task_source.parent.name
    logger.debug(f"Extracted subject id: {extracted_subject_id}")
    # we don't want to upload files from folders that don't correspond to labtracks IDs, like `sound`, or `*_test`
    if not extracted_subject_id .isdigit():
        logger.debug(
            f"Skipping {task_source} because parent folder name is not a number")
        return None
    
    if extracted_subject_id  in EXCLUDED_SUBJECT_IDS:
        logger.debug(
            f"Skipping {task_source} because subject ID is in EXCLUDED_SUBJECT_IDS")
        return None

    # if session has been already been uploaded, skip it
    session_info = npc_lims.get_session_info(task_source.stem)
    if session_info.is_uploaded:
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

    rig_storage_directory = np_codeocean_upload.CONFIG["rig_metadata_dir"]
    logger.debug(f"Rig storage directory: {rig_storage_directory}")
    add_metadata(
        task_source,
        metadata_dir,
        rig_storage_directory=rig_storage_directory,
    )

    np_tools.symlink(
        np_codeocean_upload.as_posix(task_source),
        behavior_modality_dir / task_source.name,
    )

    upload_job_contents = {
        'subject-id': extracted_subject_id,
        'acq-datetime': dynamic_routing_task.extract_session_datetime(
            task_source).strftime(ACQ_DATETIME_FORMAT),
        'project_name': 'Dynamic Routing',
        'platform': 'behavior',
        'modality0': 'behavior',
        'metadata_dir': np_config.normalize_path(metadata_dir).as_posix(),
        'modality0.source': np_config.normalize_path(
            behavior_modality_dir).as_posix(),
        'force_cloud_sync': force_cloud_sync,
    }
    
    upload_job_path = write_upload_job(
        upload_job_contents,
        np_config.normalize_path(session_dir / 'upload.csv'),
    )

    upload_service_url = np_codeocean_upload.DEV_SERVICE \
        if test else np_codeocean_upload.AIND_DATA_TRANSFER_SERVICE
    logger.debug(f"Uploading to: {upload_service_url}")
    
    np_codeocean_upload.put_csv_for_hpc_upload(
        upload_job_path,
        upload_service_url,
        "chrism@alleninstitute.org",
    )
    logger.info('Submitted to hpc upload queue')
    return upload_job_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('task_source', type=pathlib.Path)
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--force-cloud-sync', action="store_true")
    return parser.parse_args()


def main() -> None:
    upload(**vars(parse_args()))


if __name__ == '__main__':
    main()
