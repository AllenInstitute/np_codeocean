from __future__ import annotations

import csv
import pathlib
import sys
import datetime
from pathlib import Path
from typing import NamedTuple

import aind_data_transfer.jobs.s3_upload_job as s3_upload_job
import np_config
import np_logging
import np_session
import np_tools

import np_codeocean.utils as utils 

logger = np_logging.get_logger(__name__)

CONFIG = np_config.fetch('/projects/np_codeocean')

class CodeOceanSurfaceChannelUpload(NamedTuple):
    """Objects required for uploading a Mindscope Neuropixels surface channel session to CodeOcean.
    
    Paths are symlinks to files on np-exp.
    """
    session: np_session.Session
    """Session object that the paths belong to."""
    
    ephys: Path
    """Directory of symlinks to raw ephys data files on np-exp, with only one
    `recording` per `Record Node` folder."""

    job: Path
    """File containing job parameters for `aind-data-transfer`"""

class CodeOceanUpload(NamedTuple):
    """Objects required for uploading a Mindscope Neuropixels session to CodeOcean.
        Paths are symlinks to files on np-exp.
    """
    session: np_session.Session
    """Session object that the paths belong to."""

    behavior: Path
    """Directory of symlinks to files in top-level of session folder on np-exp,
    plus all files in `exp` subfolder, if present."""
    
    ephys: Path
    """Directory of symlinks to raw ephys data files on np-exp, with only one
    `recording` per `Record Node` folder."""

    job: Path
    """File containing job parameters for `aind-data-transfer`"""
    

def create_ephys_symlinks(session: np_session.Session, dest: Path) -> None:
    """Create symlinks in `dest` pointing to raw ephys data files on np-exp, with only one
    `recording` per `Record Node` folder (the largest, if multiple found).
    
    Relative paths are preserved, so `dest` will essentially be a merge of
    _probeABC / _probeDEF folders.
    
    Top-level items other than `Record Node *` folders are excluded.
    """
    logger.info(f'Creating symlinks to raw ephys data files in {session.npexp_path}...')
    for abs_path, rel_path in np_tools.get_filtered_ephys_paths_relative_to_record_node_parents(session.npexp_path):
        if not abs_path.is_dir():
            np_tools.symlink(abs_path, dest / rel_path)
    logger.debug(f'Finished creating symlinks to raw ephys data files in {session.npexp_path}')

         
def create_behavior_symlinks(session: np_session.Session, dest: Path) -> None:
    """Create symlinks in `dest` pointing to files in top-level of session
    folder on np-exp, plus all files in `exp` subfolder, if present.
    """
    logger.info(f'Creating symlinks in {dest} to files in {session.npexp_path}...')
    for src in session.npexp_path.glob('*'):
        if not src.is_dir():
            np_tools.symlink(src, dest / src.relative_to(session.npexp_path))
    logger.debug(f'Finished creating symlinks to top-level files in {session.npexp_path}')

    if not (session.npexp_path / 'exp').exists():
        return
    
    for src in (session.npexp_path / 'exp').rglob('*'):
        if not src.is_dir():
            np_tools.symlink(src, dest / src.relative_to(session.npexp_path))
    logger.debug(f'Finished creating symlinks to files in {session.npexp_path / "exp"}')


def get_ephys_upload_csv_for_session(session: np_session.Session, ephys: Path, behavior: Path) -> dict[str, str | int]:
    return {
        'data-source': np_config.normalize_path(ephys).as_posix(),
        'behavior-dir': np_config.normalize_path(behavior).as_posix(),
        's3-bucket': CONFIG['s3-bucket'],
        'subject-id': str(session.mouse),
        'experiment-type': 'ecephys',
        'modality': 'ECEPHYS',
        'acq-date': f'{session.date:%Y-%m-%d}',
        'acq-time': f'{session.start:%H-%M-%S}',
        'aws-param-store-name': CONFIG['aws-param-store-name'],
    } # type: ignore

def get_surface_channel_start_time(session: np_session.Session) -> datetime.datetime:
    sync_messages_path = tuple(session.npexp_path.glob('*/*/*/sync_messages.txt'))
    if not sync_messages_path:
        raise ValueError(f'No sync messages txt found for surface channel session {session}')
    sync_messages_path = sync_messages_path[0]

    with open(sync_messages_path, 'r') as f:
        software_time_line = f.readlines()[0]

    timestamp_value = float(software_time_line[software_time_line.index(':')+2:].strip())
    timestamp = datetime.datetime.fromtimestamp(timestamp_value / 1e3)
    return timestamp
        
def get_surface_channel_ephys_upload_csv_for_session(session: np_session.Session, ephys:Path) -> dict[str, str | int]:
    return {
        'data-source': np_config.normalize_path(ephys).as_posix(),
        's3-bucket': CONFIG['s3-bucket'],
        'subject-id': str(session.mouse),
        'experiment-type': 'ecephys',
        'modality': 'ECEPHYS',
        'acq-date': f'{session.date:%Y-%m-%d}',
        'acq-time': f'{get_surface_channel_start_time(session):%H-%M-%S}_surface_channels', # probably won't work but worth a shot
        'aws-param-store-name': CONFIG['aws-param-store-name'],
    } # type: ignore

def create_upload_job(session: np_session.Session, job: Path, ephys: Path, behavior: Path) -> None:
    logger.info(f'Creating upload job file {job} for session {session}...')
    _csv = get_ephys_upload_csv_for_session(session, ephys, behavior)
    with open(job, 'w') as f:
        w = csv.writer(f)
        w.writerow(_csv.keys())
        w.writerow(_csv.values()) 

def create_surface_channel_upload_job(session: np_session.Session, job: Path, ephys: Path) -> None:
    logger.info(f'Creating upload job file {job} for session {session}...')
    _csv = get_surface_channel_ephys_upload_csv_for_session(session, ephys)
    with open(job, 'w') as f:
        w = csv.writer(f)
        w.writerow(_csv.keys())
        w.writerow(_csv.values()) 

def create_codeocean_upload(session: str | int | np_session.Session) -> CodeOceanUpload:
    """Create directories of symlinks to np-exp files with correct structure
    for upload to CodeOcean.
    
    - only one `recording` per `Record Node` folder (largest if multiple found)
    - job file for feeding into `aind-data-transfer`
    """
    
    session = np_session.Session(session)
    root = np_session.NPEXP_PATH / 'codeocean' / session.folder
    logger.debug(f'Created directory {root} for CodeOcean upload')
    
    upload = CodeOceanUpload(
        session = session, 
        behavior = np_config.normalize_path(root / 'behavior'),
        ephys = np_config.normalize_path(root / 'ephys'),
        job = np_config.normalize_path(root / 'upload.csv'),
        )

    create_ephys_symlinks(upload.session, upload.ephys)
    create_behavior_symlinks(upload.session, upload.behavior)
    create_upload_job(upload.session, upload.job, upload.ephys, upload.behavior)    
    return upload

def create_codeocean_surface_channel_upload(session: str | int | np_session.Session) -> CodeOceanSurfaceChannelUpload:
    session = np_session.Session(session)
    root = np_session.NPEXP_PATH / 'codeocean' / f'{session.folder}_surface_channels'
    logger.debug(f'Created directory {root} for Surface Channel CodeOcean upload')

    upload = CodeOceanSurfaceChannelUpload(
        session=session,
        ephys = np_config.normalize_path(root / 'ephys'),
        job = np_config.normalize_path(root / 'upload.csv'),
    )
    create_ephys_symlinks(upload.session, upload.ephys)
    create_surface_channel_upload_job(upload.session, upload.job, upload.ephys)    
    return upload


def upload_session(session: str | int | pathlib.Path | np_session.Session) -> None:
    utils.ensure_credentials()

    if 'surface' in str(session):
        upload = create_codeocean_surface_channel_upload(str(session))
    else:
        upload = create_codeocean_upload(str(session))
        
    np_logging.web('np_codeocean').info(f'Uploading {upload.session}')
    s3_upload_job.GenericS3UploadJobList(["--jobs-csv-file", upload.job.as_posix()]).run_job()
    np_logging.web('np_codeocean').info(f'Finished uploading {upload.session}')
    
def main() -> None:
    upload_session(sys.argv[1])
    
if __name__ == '__main__':
    session = np_session.Session(r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_660023_20230808_surface_channels")
    get_surface_channel_start_time(session)
    #main()