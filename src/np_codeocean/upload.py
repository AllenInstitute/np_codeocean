from __future__ import annotations

import csv
import pathlib
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import aind_data_transfer.jobs.s3_upload_job as s3_upload_job
import np_config
import np_logging
import np_session
import np_tools


logger = np_logging.get_logger(__name__)

CONFIG = np_config.fetch('/projects/np_codeocean')

class CodeOceanUpload(NamedTuple):
    """Objects required for uploading a Mindscope Neuropixels session to CodeOcean.
    
    Paths are in tempdirs, with symlinks to files on np-exp.
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
    """
    logger.info(f'Creating symlinks to raw ephys data files in {session.npexp_path}...')
    ephys_subfolders = np_tools.get_raw_ephys_subfolders(session.npexp_path)
    logger.debug(f'Found {len(ephys_subfolders)} ephys subfolders: {ephys_subfolders}')
    for ephys_folder in ephys_subfolders:
        superfluous_recording_dirs: tuple[pathlib.Path, ...] = tuple(
            _.parent for _ in np_tools.get_superfluous_oebin_paths(ephys_folder)
        )
        logger.debug(f'Found {len(superfluous_recording_dirs)} superfluous recording dirs to exclude: {superfluous_recording_dirs}')
        for src in ephys_folder.rglob('*'):
            is_superfluous_path = any(_ in src.parents for _ in superfluous_recording_dirs)
            if is_superfluous_path:
                continue
            if src.is_file():
                np_tools.symlink(src, dest / src.relative_to(ephys_folder))
    logger.debug(f'Finished creating symlinks to raw ephys data files in {session.npexp_path}')


def create_behavior_symlinks(session: np_session.Session, dest: Path) -> None:
    """Create symlinks in `dest` pointing to files in top-level of session
    folder on np-exp, plus all files in `exp` subfolder, if present.
    """
    logger.info(f'Creating symlinks in {dest} to files in {session.npexp_path}...')
    for src in session.npexp_path.glob('*'):
        if src.is_file():
            np_tools.symlink(src, dest / src.relative_to(session.npexp_path))
    logger.debug(f'Finished creating symlinks to top-level files in {session.npexp_path}')

    if not (session.npexp_path / 'exp').exists():
        return
    
    for src in (session.npexp_path / 'exp').rglob('*'):
        if src.is_file():
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

def create_upload_job(session: np_session.Session, job: Path, ephys: Path, behavior: Path) -> None:
    logger.info(f'Creating upload job file {job} for session {session}...')
    _csv = get_ephys_upload_csv_for_session(session, ephys, behavior)
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
    
    root = Path(tempfile.mkdtemp())
    logger.debug(f'Created temporary directory {root} for CodeOcean upload')
    
    upload = CodeOceanUpload(
        session = np_session.Session(session), 
        behavior = np_config.normalize_path(root / 'behavior'),
        ephys = np_config.normalize_path(root / 'ephys'),
        job = np_config.normalize_path(root / 'upload.csv'),
        )

    create_ephys_symlinks(upload.session, upload.ephys)
    create_behavior_symlinks(upload.session, upload.behavior)
    create_upload_job(upload.session, upload.job, upload.ephys, upload.behavior)    
    return upload


def main()-> None:
    upload = create_codeocean_upload(sys.argv[1])
    np_logging.web('np_codeocean').info(f'Uploading {upload.session}')
    s3_upload_job.GenericS3UploadJobList(["--jobs-csv-file", upload.job.as_posix()]).run_job()
    np_logging.web('np_codeocean').info(f'Finished uploading {upload.session}')
    

if __name__ == '__main__':
    np_logging.getLogger()
    main()