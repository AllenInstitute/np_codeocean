from __future__ import annotations

import argparse
import dataclasses
import datetime
import doctest
import pathlib
from collections.abc import Iterable

import np_config
import np_logging
import np_session
import np_tools

import np_codeocean.utils as utils

logger = np_logging.get_logger(__name__)

@dataclasses.dataclass
class CodeOceanUpload:
    """Objects required for uploading a Mindscope Neuropixels session to CodeOcean.
        Paths are symlinks to files on np-exp.
    """
    session: np_session.Session
    """Session object that the paths belong to."""
    
    platform: utils.AINDPlatform
    """The 'platform' in the Neural Dynamics data schema language (effectively the rig
    type, which determines the processing pipeline the data follows).
     
    Our rules are:
    - if it ran in a behavior box: `behavior`
    - anything else: `ecephys`
    
    This means there will be behavior-only sessions that ran on NP-rigs
    without ephys data (habs, opto experiments etc.), that will be uploaded as
    `ecephys` platform data.
    """

    behavior: pathlib.Path | None
    """Directory of symlinks to files in top-level of session folder on np-exp,
    plus all files in `exp` and `qc` subfolders, if present. Excludes behavior video files
    and video info jsons."""
    
    behavior_videos: pathlib.Path | None    
    """Directory of symlinks to behavior video files and video info jsons in
    top-level of session folder on np-exp."""
    
    ephys: pathlib.Path | None
    """Directory of symlinks to raw ephys data files on np-exp, with only one
    `recording` per `Record Node` folder."""

    aind_metadata: pathlib.Path | None
    """Directory of symlinks to aind metadata json files in top-level of session folder 
    on np-exp."""

    job: pathlib.Path
    """File containing job parameters for `aind-data-transfer`"""

    force_cloud_sync: bool = False
    """If True, re-upload and re-make raw asset even if data exists on S3."""

    @property
    def project_name(self) -> str:
        if isinstance(self.session, np_session.PipelineSession):
            return "OpenScope"
        return "Dynamic Routing"



def attempt_create_aind_metadata_symlinks(session: np_session.Session, dest: pathlib.Path) -> bool:
    """
    Create symlinks in `dest` pointing to aind metadata json files from the root directory
    on np-exp. Returns True if any metadata files are found in np-exp and the `aind_metadata`
    folder is created.
    """
    has_metadata_files = False
    for src in session.npexp_path.glob('*'):
        for metadata_file_name in ('session', 'data_description', 'procedures', 'processing', 'rig', 'subject'):
            if src.stem + src.suffix == f'{metadata_file_name}.json':
                np_tools.symlink(utils.ensure_posix(src), dest / src.relative_to(session.npexp_path))    
                has_metadata_files = True
    if has_metadata_files:
        logger.debug(f'Finished creating symlinks to aind metadata files in {session.npexp_path}')
    else:
        logger.debug(f'No metadata files found in {session.npexp_path}; No symlinks for metadata were made')
    return has_metadata_files


def create_ephys_symlinks(session: np_session.Session, dest: pathlib.Path, 
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
            np_tools.symlink(utils.ensure_posix(abs_path), dest / rel_path)
    logger.debug(f'Finished creating symlinks to raw ephys data files in {root_path}')
    utils.cleanup_ephys_directories(dest)


def create_behavior_symlinks(session: np_session.Session, dest: pathlib.Path | None) -> None:
    """Create symlinks in `dest` pointing to files in top-level of session
    folder on np-exp, plus all files in `exp` subfolder, if present.
    """
    if dest is None: 
        logger.debug(f"No behavior folder supplied for {session}")
        return
    subfolder_names = ('exp', 'qc')
    logger.info(f'Creating symlinks in {dest} to files in {session.npexp_path}...')
    for src in session.npexp_path.glob('*'):
        if not src.is_dir() and not utils.is_behavior_video_file(src):
            np_tools.symlink(utils.ensure_posix(src), dest / src.relative_to(session.npexp_path))
    logger.debug(f'Finished creating symlinks to top-level files in {session.npexp_path}')

    for name in subfolder_names:
        subfolder = session.npexp_path / name
        if not subfolder.exists():
            continue
        for src in subfolder.rglob('*'):
            if not src.is_dir():
                np_tools.symlink(utils.ensure_posix(src), dest / src.relative_to(session.npexp_path))
        logger.debug(f'Finished creating symlinks to {name!r} files')


def create_behavior_videos_symlinks(session: np_session.Session, dest: pathlib.Path | None) -> None:
    """Create symlinks in `dest` pointing to MVR video files and info jsons in top-level of session
    folder on np-exp.
    """
    if dest is None: 
        logger.debug(f"No behavior_videos folder supplied for {session}")
        return
    logger.info(f'Creating symlinks in {dest} to files in {session.npexp_path}...')
    for src in session.npexp_path.glob('*'):
        if utils.is_behavior_video_file(src):
            np_tools.symlink(utils.ensure_posix(src), dest / src.relative_to(session.npexp_path))
    logger.debug(f'Finished creating symlinks to behavior video files in {session.npexp_path}')


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
    >>> utils.is_surface_channel_recording(path)
    True
    >>> upload = create_codeocean_upload(path)
    >>> ephys_upload_csv = get_upload_csv_for_session(upload)
    >>> ephys_upload_csv['modality0.source']
    '//allen/programs/mindscope/workgroups/np-exp/codeocean/DRpilot_690706_20231129_surface_channels/ephys'
    >>> ephys_upload_csv.keys()
    dict_keys(['project_name', 'platform', 'subject-id', 'force_cloud_sync', 'modality0', 'modality0.source', 'acq-datetime'])
    """
    params = {
        'project_name': upload.project_name,
        'platform': upload.platform,
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
    
    if upload.aind_metadata:
        params['metadata_dir'] = upload.aind_metadata.as_posix()
            
    if utils.is_surface_channel_recording(upload.session.npexp_path.as_posix()):
        date = datetime.datetime(upload.session.date.year, upload.session.date.month, upload.session.date.day)
        session_date_time = date.combine(upload.session.date, get_surface_channel_start_time(upload.session).time())
        params['acq-datetime'] = f'{session_date_time.strftime(utils.ACQ_DATETIME_FORMAT)}'
    else:
        params['acq-datetime'] = f'{upload.session.start.strftime(utils.ACQ_DATETIME_FORMAT)}'
    
    return params # type: ignore


def is_ephys_session(session: np_session.Session) -> bool:
    return bool(next(session.npexp_path.rglob('settings.xml'), None))


def create_codeocean_upload(
    session: str | int | np_session.Session,
    recording_dirs: Iterable[str] | None = None,
    force_cloud_sync: bool = False,
    codeocean_root: pathlib.Path = np_session.NPEXP_PATH / 'codeocean',
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
    platform: utils.AINDPlatform = 'ecephys' # all session-type uploads with a folder of data are ecephys platform; behavior platform is for behavior-box sessions

    if utils.is_surface_channel_recording(str(session)):
        session = np_session.Session(session)
        if not utils.is_surface_channel_recording(session.npexp_path.name):
            # manually assign surface channel path 
            session = np_session.Session(session.npexp_path.parent / f'{session.folder}_surface_channels')
            assert session.npexp_path.exists(), f"Surface channel path {session.npexp_path} does not exist in same folder as main session recording"
        root = codeocean_root / f'{session.folder}_surface_channels'
        behavior = None
        behavior_videos = None
    else:
        session = np_session.Session(session)
        root = codeocean_root / session.folder
        behavior = np_config.normalize_path(root / 'behavior')
        behavior_videos = behavior.with_name('behavior-videos')

    logger.debug(f'Created directory {root} for CodeOcean upload')

    logger.info('Attempting to create sub directory for AIND metadata jsons..')
    metadata_path = np_config.normalize_path(root / 'aind_metadata')
    has_metadata: bool = attempt_create_aind_metadata_symlinks(session, metadata_path)

    return CodeOceanUpload(
        session = session, 
        behavior = behavior,
        behavior_videos = behavior_videos,
        ephys = np_config.normalize_path(root / 'ephys') if is_ephys_session(session) else None,
        aind_metadata = metadata_path if has_metadata else None,
        job = np_config.normalize_path(root / 'upload.csv'),
        force_cloud_sync=force_cloud_sync,
        platform=platform,
    )

def upload_session(
    session: str | int | pathlib.Path | np_session.Session, 
    recording_dirs: Iterable[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    test: bool = False,
    hpc_upload_job_email: str = utils.HPC_UPLOAD_JOB_EMAIL,
) -> None:
    codeocean_root = np_session.NPEXP_PATH / 'codeocean' if not test \
        else np_session.NPEXP_PATH / 'codeocean-dev'
    logger.debug(f'Codeocean root: {codeocean_root}')
    upload = create_codeocean_upload(
        str(session),
        codeocean_root=codeocean_root,
        recording_dirs=recording_dirs,
        force_cloud_sync=force
    )
    if dry_run:
        logger.info(f'Dry run. Not submitting {upload.session} to hpc upload queue. dry_run={dry_run}, upload={upload}')
        return
    if upload.ephys:
        create_ephys_symlinks(upload.session, upload.ephys, recording_dirs=recording_dirs)
    if upload.behavior:
        create_behavior_symlinks(upload.session, upload.behavior)
    if upload.behavior_videos:
        create_behavior_videos_symlinks(upload.session, upload.behavior_videos)
    csv_content: dict = get_upload_csv_for_session(upload)
    utils.write_upload_csv(csv_content, upload.job)
    np_logging.web('np_codeocean').info(f'Submitting {upload.session} to hpc upload queue')
    utils.put_csv_for_hpc_upload(
        csv_path=upload.job,
        upload_service_url=utils.DEV_SERVICE if test else utils.AIND_DATA_TRANSFER_SERVICE,
        hpc_upload_job_email=hpc_upload_job_email,
        dry_run=dry_run,
    )
    logger.debug(f'Submitted {upload.session} to hpc upload queue')
    
    if (is_split_recording := 
        recording_dirs is not None 
        and len(tuple(recording_dirs)) > 1 
        and isinstance(recording_dirs, str)
    ):
        logger.warning(f"Split recording {upload.session} will need to be sorted manually with `CONCAT=True`")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a session to CodeOcean")
    parser.add_argument('session', help="session ID (lims or np-exp foldername) or path to session folder")
    parser.add_argument('--force', action='store_true', help="enable `force_cloud_sync` option, re-uploading and re-making raw asset even if data exists on S3")
    parser.add_argument('--test', action='store_true', help="use the test-upload service, uploading to the test CodeOcean server instead of the production server")
    parser.add_argument('recording_dirs', nargs='*', type=list, help="[optional] specific recording directories to upload - for use with split recordings only.")
    parser.add_argument('--dry-run', action='store_true', help="Create upload job but do not submit to hpc upload queue.")
    return parser.parse_args()

if __name__ == '__main__':
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE),
    )