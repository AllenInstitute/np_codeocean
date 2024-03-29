from __future__ import annotations

import argparse
import contextlib
import csv
import json
import pathlib
import datetime
from pathlib import Path
from typing import NamedTuple
from collections.abc import Iterable

import np_config
import np_logging
import np_session
import npc_session
import np_tools
import doctest
import numpy as np
import polars as pl
import requests
from pydantic import ValidationError # may be returned from aind-data-transfer-service

logger = np_logging.get_logger(__name__)

CONFIG = np_config.fetch('/projects/np_codeocean')
AIND_DATA_TRANSFER_SERVICE = "http://aind-data-transfer-service"


class CodeOceanUpload(NamedTuple):
    """Objects required for uploading a Mindscope Neuropixels session to CodeOcean.
        Paths are symlinks to files on np-exp.
    """
    session: np_session.Session
    """Session object that the paths belong to."""

    behavior: Path | None
    """Directory of symlinks to files in top-level of session folder on np-exp,
    plus all files in `exp` and `qc` subfolders, if present. Excludes behavior video files
    and video info jsons."""
    
    behavior_videos: Path | None    
    """Directory of symlinks to behavior video files and video info jsons in
    top-level of session folder on np-exp."""
    
    ephys: Path | None
    """Directory of symlinks to raw ephys data files on np-exp, with only one
    `recording` per `Record Node` folder."""

    job: Path
    """File containing job parameters for `aind-data-transfer`"""

    force_cloud_sync: bool = False
    """If True, re-upload and re-make raw asset even if data exists on S3."""
    
def as_posix(path: pathlib.Path) -> str:
    return path.as_posix()[1:]

def create_ephys_symlinks(session: np_session.Session, dest: Path, 
                          recording_dirs: Iterable[str] | None = None) -> None:
    """Create symlinks in `dest` pointing to raw ephys data files on np-exp, with only one
    `recording` per `Record Node` folder (the largest, if multiple found).
    
    Relative paths are preserved, so `dest` will essentially be a merge of
    _probeABC / _probeDEF folders.
    
    Top-level items other than `Record Node *` folders are excluded.
    """
    root_path = session.npexp_path
    if isinstance(session, np_session.PipelineSession) and session.lims_path is not None:
        # if ephys has been uploaded to lims, use lims path, as large raw data may have
        # been deleted from np-exp
        if any(
            np_tools.get_filtered_ephys_paths_relative_to_record_node_parents(
                session.npexp_path, specific_recording_dir_names=recording_dirs
            )
        ):
            root_path = session.lims_path
    logger.info(f'Creating symlinks to raw ephys data files in {root_path}...')
    for abs_path, rel_path in np_tools.get_filtered_ephys_paths_relative_to_record_node_parents(
        root_path, specific_recording_dir_names=recording_dirs
        ):
        if not abs_path.is_dir():
            np_tools.symlink(as_posix(abs_path), dest / rel_path)
    logger.debug(f'Finished creating symlinks to raw ephys data files in {root_path}')

def is_behavior_video_file(path: Path) -> bool:
    if path.is_dir() or path.suffix not in ('.mp4', '.avi', '.json'):
        return False
    with contextlib.suppress(ValueError):
        _ = npc_session.extract_mvr_camera_name(path.as_posix())
        return True
    return False

def create_behavior_symlinks(session: np_session.Session, dest: Path | None) -> None:
    """Create symlinks in `dest` pointing to files in top-level of session
    folder on np-exp, plus all files in `exp` subfolder, if present.
    """
    if dest is None: 
        logger.debug(f"No behavior folder supplied for {session}")
        return
    subfolder_names = ('exp', 'qc')
    logger.info(f'Creating symlinks in {dest} to files in {session.npexp_path}...')
    for src in session.npexp_path.glob('*'):
        if not src.is_dir() and not is_behavior_video_file(src):
            np_tools.symlink(as_posix(src), dest / src.relative_to(session.npexp_path))
    logger.debug(f'Finished creating symlinks to top-level files in {session.npexp_path}')

    for name in subfolder_names:
        subfolder = session.npexp_path / name
        if not subfolder.exists():
            continue
        for src in subfolder.rglob('*'):
            if not src.is_dir():
                np_tools.symlink(as_posix(src), dest / src.relative_to(session.npexp_path))
        logger.debug(f'Finished creating symlinks to {name!r} files')

def create_behavior_videos_symlinks(session: np_session.Session, dest: Path | None) -> None:
    """Create symlinks in `dest` pointing to MVR video files and info jsons in top-level of session
    folder on np-exp.
    """
    if dest is None: 
        logger.debug(f"No behavior_videos folder supplied for {session}")
        return
    logger.info(f'Creating symlinks in {dest} to files in {session.npexp_path}...')
    for src in session.npexp_path.glob('*'):
        if is_behavior_video_file(src):
            np_tools.symlink(as_posix(src), dest / src.relative_to(session.npexp_path))
    logger.debug(f'Finished creating symlinks to behavior video files in {session.npexp_path}')
    
def is_surface_channel_recording(path_name: str) -> bool:
    """
    >>> session = np_session.Session("//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_690706_20231129_surface_channels")
    >>> is_surface_channel_recording(session.npexp_path.as_posix())
    True
    """
    return 'surface_channels' in path_name.lower()

def get_surface_channel_start_time(session: np_session.Session) -> datetime.datetime:
    """
    >>> session = np_session.Session("//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_690706_20231129_surface_channels")
    >>> get_surface_channel_start_time(session)
    datetime.datetime(2023, 11, 29, 14, 56, 25, 219000)
    """
    sync_messages_paths = tuple(session.npexp_path.glob('*/*/*/sync_messages.txt'))
    if not sync_messages_paths:
        raise ValueError(f'No sync messages txt found for surface channel session {session}')
    sync_messages_path = sync_messages_paths[0]

    with open(sync_messages_path, 'r') as f:
        software_time_line = f.readlines()[0]

    timestamp_value = float(software_time_line[software_time_line.index(':')+2:].strip())
    timestamp = datetime.datetime.fromtimestamp(timestamp_value / 1e3)
    return timestamp

def get_upload_csv_for_session(upload: CodeOceanUpload) -> dict[str, str | int | bool]:
    """
    >>> path = "//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_690706_20231129_surface_channels"
    >>> is_surface_channel_recording(path)
    True
    >>> upload = create_codeocean_upload(path)
    >>> ephys_upload_csv = get_upload_csv_for_session(upload)
    >>> ephys_upload_csv['modality0.source']
    '//allen/programs/mindscope/workgroups/np-exp/codeocean/DRpilot_690706_20231129_surface_channels/ephys'
    >>> ephys_upload_csv.keys()
    dict_keys(['platform', 'subject-id', 'force_cloud_sync', 'modality0', 'modality0.source', 'acq-datetime'])
    """
    params = {
        'platform': 'ecephys',
        'subject-id': str(upload.session.mouse),
        'force_cloud_sync': upload.force_cloud_sync,
    }
    idx = 0
    for modality_name, attr_name in {
        'ecephys': 'ephys',
        'behavior': 'behavior',
        'behavior-videos': 'behavior_videos',
    }.items():
        if getattr(upload, attr_name) is not None:
            params[f'modality{idx}'] = modality_name
            params[f'modality{idx}.source'] = np_config.normalize_path(getattr(upload, attr_name)).as_posix()
            idx += 1
            
    if is_surface_channel_recording(upload.session.npexp_path.as_posix()):
        date = datetime.datetime(upload.session.date.year, upload.session.date.month, upload.session.date.day)
        session_date_time = date.combine(upload.session.date, get_surface_channel_start_time(upload.session).time())
        params['acq-datetime'] = f'{session_date_time.strftime("%Y-%m-%d %H:%M:%S")}'
    else:
        params['acq-datetime'] = f'{upload.session.start.strftime("%Y-%m-%d %H:%M:%S")}'
    
    return params


def is_in_hpc_upload_queue(csv_path: pathlib.Path) -> bool:
    """Check if an upload job has been submitted to the hpc upload queue.
    
    - currently assumes one job per csv
    - does not check status (job may be FINISHED rather than RUNNING)
    
    >>> is_in_hpc_upload_queue("//allen/programs/mindscope/workgroups/np-exp/codeocean/DRpilot_664851_20231114/upload.csv")
    False
    """
    # get subject-id, acq-datetime from csv
    df = pl.read_csv(csv_path, eol_char='\r')
    for col in df.get_columns():
        if col.name.startswith('subject') and col.name.endswith('id'):
            subject = npc_session.SubjectRecord(col[0])
            continue
        if col.name.startswith('acq') and 'datetime' in col.name.lower():
            dt = npc_session.DatetimeRecord(col[0])
            continue
    partial_session_id = f"{subject}_{dt.replace(' ', '_').replace(':', '-')}"
    
    jobs_response = requests.get(f"{AIND_DATA_TRANSFER_SERVICE}/jobs")
    jobs_response.raise_for_status()
    return partial_session_id in jobs_response.content.decode()
    
def put_csv_for_hpc_upload(csv_path: pathlib.Path) -> None:
    """Submit a single job upload csv to the aind-data-transfer-service, for
    upload to S3 on the hpc.
    
    - gets validated version of csv
    - checks session is not already being uploaded
    - submits csv via http request
    """
    def _raise_for_status(response: requests.Response) -> None:
        """pydantic validation errors are returned as strings that can be eval'd
        to get the real error class + message."""
        if response.status_code != 200:
            try:
                x = response.json()['data']['errors']
                import pdb; pdb.set_trace()
            except (KeyError, IndexError, requests.exceptions.JSONDecodeError, SyntaxError) as exc1:
                try:
                    response.raise_for_status()
                except requests.exceptions.HTTPError as exc2:
                    raise exc2 from exc1
                
    with open(csv_path, 'rb') as f:
        validate_csv_response = requests.post(
            url=f"{AIND_DATA_TRANSFER_SERVICE}/api/validate_csv", 
            files=dict(file=f),
            )
    _raise_for_status(validate_csv_response)
    
    if is_in_hpc_upload_queue(csv_path):
        logger.warning(f"Job already submitted for {csv_path}")
        return
    
    post_csv_response = requests.post(
        url=f"{AIND_DATA_TRANSFER_SERVICE}/api/submit_hpc_jobs", 
        json=dict(
            jobs=[
                    dict(
                        hpc_settings=json.dumps({"time_limit": 60 * 15, "mail_user": "arjun.sridhar@alleninstitute.org"}),
                        upload_job_settings=validate_csv_response.json()["data"]["jobs"][0],
                        script="",
                    )
                ]
        ),
    )
    _raise_for_status(post_csv_response)

def is_ephys_session(session: np_session.Session) -> bool:
    return bool(next(session.npexp_path.rglob('settings.xml'), None))

def create_upload_job(upload: CodeOceanUpload) -> None:
    logger.info(f'Creating upload job file {upload.job} for session {upload.session}...')
    job: dict = get_upload_csv_for_session(upload)
    with open(upload.job, 'w') as f:
        w = csv.writer(f, lineterminator='')
        w.writerow(job.keys())
        w.writerow('\n')

        w.writerow(job.values()) 

def create_codeocean_upload(session: str | int | np_session.Session, 
                            recording_dirs: Iterable[str] | None = None,
                            force_cloud_sync: bool = False,
                            ) -> CodeOceanUpload:
    """Create directories of symlinks to np-exp files with correct structure
    for upload to CodeOcean.
    
    - only one `recording` per `Record Node` folder (largest if multiple found)
    - job file for feeding into `aind-data-transfer`
    
    >>> upload = create_codeocean_upload("//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_690706_20231129_surface_channels")
    >>> upload.behavior is None
    True
    >>> upload.ephys.exists()
    True
    """

    if is_surface_channel_recording(str(session)):
        session = np_session.Session(session)
        if not is_surface_channel_recording(session.npexp_path.name):
            # manually assign surface channel path 
            session = np_session.Session(session.npexp_path.parent / f'{session.folder}_surface_channels')
            assert session.npexp_path.exists(), f"Surface channel path {session.npexp_path} does not exist in same folder as main session recording"
        root = np_session.NPEXP_PATH / 'codeocean' / f'{session.folder}_surface_channels'
        behavior = None
        behavior_videos = None
    else:
        session = np_session.Session(session)
        root = np_session.NPEXP_PATH / 'codeocean' / session.folder
        behavior = np_config.normalize_path(root / 'behavior')
        behavior_videos = behavior.with_name('behavior-videos')
        
    logger.debug(f'Created directory {root} for CodeOcean upload')
    
    upload = CodeOceanUpload(
        session = session, 
        behavior = behavior,
        behavior_videos = behavior_videos,
        ephys = np_config.normalize_path(root / 'ephys') if is_ephys_session(session) else None,
        job = np_config.normalize_path(root / 'upload.csv'),
        force_cloud_sync=force_cloud_sync,
        )

    create_ephys_symlinks(upload.session, upload.ephys, recording_dirs=recording_dirs)
    create_behavior_symlinks(upload.session, upload.behavior)
    create_behavior_videos_symlinks(upload.session, upload.behavior_videos)
    create_upload_job(upload)    
    return upload

def upload_session(session: str | int | pathlib.Path | np_session.Session, 
                   recording_dirs: Iterable[str] | None = None,
                   force: bool = False,
                   ) -> None:
    upload = create_codeocean_upload(str(session), recording_dirs=recording_dirs, force_cloud_sync=force)
    np_logging.web('np_codeocean').info(f'Submitting {upload.session} to hpc upload queue')
    put_csv_for_hpc_upload(upload.job)
    logger.debug(f'Submitted {upload.session} to hpc upload queue')
    
    if (is_split_recording := 
        recording_dirs is not None 
        and len(tuple(recording_dirs)) > 1 
        and isinstance(recording_dirs, str)
    ):
        logger.warning(f"Split recording {upload.session} will need to be sorted manually with `CONCAT=True`")

    
def main() -> None:
    upload_session(**vars(parse_args()))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a session to CodeOcean")
    parser.add_argument('session', help="session ID (lims or np-exp foldername) or path to session folder")
    parser.add_argument('--force', action='store_true', help="enable `force_cloud_sync` option, re-uploading and re-making raw asset even if data exists on S3")
    parser.add_argument('recording_dirs', nargs='*', type=list, help="[optional] specific recording directories to upload - for use with split recordings only.")
    return parser.parse_args()

if __name__ == '__main__':
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE),
    )