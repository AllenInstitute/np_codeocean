from __future__ import annotations

import dataclasses
import datetime
import doctest
import pathlib
from collections.abc import Iterable
import shutil
import time
from typing import Any

from aind_codeocean_pipeline_monitor.models import PipelineMonitorSettings
import np_config
import np_logging
import np_session
import np_tools

import np_codeocean.utils as utils
import typing_extensions
from aind_data_schema_models.modalities import Modality

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

    @property
    def root(self) -> pathlib.Path:
        for attr in (self.behavior, self.behavior_videos, self.ephys, self.aind_metadata):
            if attr is not None:
                return attr.parent
        raise ValueError(f"No upload directories assigned to {self!r}")

def create_aind_metadata_symlinks(upload: CodeOceanUpload) -> bool:
    """
    Create symlinks in `dest` pointing to aind metadata json files from the root directory
    on np-exp. Returns True if any metadata files are found in np-exp and the `aind_metadata`
    folder is created.
    """
    has_metadata_files = False
    for src in upload.session.npexp_path.glob('*'):
        if src.stem in utils.AIND_METADATA_NAMES:
            np_tools.symlink(src, upload.aind_metadata / src.name)
            has_metadata_files = True
    if has_metadata_files:
        logger.debug(f'Finished creating symlinks to aind metadata files in {upload.session.npexp_path}')
    else:
        logger.debug(f'No metadata files found in {upload.session.npexp_path}; No symlinks for metadata were made')
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
            np_tools.symlink(abs_path, dest / rel_path)
    logger.debug(f'Finished creating symlinks to raw ephys data files in {root_path}')
    utils.cleanup_ephys_symlinks(dest)


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
            np_tools.symlink(src, dest / src.relative_to(session.npexp_path))
    logger.debug(f'Finished creating symlinks to top-level files in {session.npexp_path}')

    for name in subfolder_names:
        subfolder = session.npexp_path / name
        if not subfolder.exists():
            continue
        for src in subfolder.rglob('*'):
            if not src.is_dir():
                np_tools.symlink(src, dest / src.relative_to(session.npexp_path))
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
            np_tools.symlink(src, dest / src.relative_to(session.npexp_path))
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


def get_upload_params_from_session(upload: CodeOceanUpload) -> dict[str, Any]:
    """
    >>> path = "//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_690706_20231129_surface_channels"
    >>> utils.is_surface_channel_recording(path)
    True
    >>> upload = create_codeocean_upload(path)
    >>> ephys_upload_params = get_upload_params_from_session(upload)
    >>> ephys_upload_params['modalities']['ecephys']
    '//allen/programs/mindscope/workgroups/np-exp/codeocean/DRpilot_690706_20231129_surface_channels/ephys'
    >>> ephys_upload_params.keys()
    dict_keys(['project_name', 'platform', 'subject_id', 'force_cloud_sync', 'modalities', 'acq_datetime'])
    """
    params = {
        'project_name': upload.project_name,
        'platform': upload.platform,
        'subject_id': str(upload.session.mouse),
        'force_cloud_sync': upload.force_cloud_sync,
    }
    modalities = dict() # {modality_abbr: input_source}
    for modality_abbr, attr_name in {
        Modality.ECEPHYS.abbreviation: 'ephys',
        Modality.BEHAVIOR.abbreviation: 'behavior',
        Modality.BEHAVIOR_VIDEOS.abbreviation: 'behavior_videos',
    }.items():
        if getattr(upload, attr_name) is not None:
            modalities[modality_abbr] = np_config.normalize_path(getattr(upload, attr_name)).as_posix()
    params['modalities'] = modalities
    
    if upload.aind_metadata:
        params['metadata_dir'] = upload.aind_metadata.as_posix()
            
    if utils.is_surface_channel_recording(upload.session.npexp_path.as_posix()):
        date = datetime.datetime(upload.session.date.year, upload.session.date.month, upload.session.date.day)
        params['acq_datetime'] = date.combine(upload.session.date, get_surface_channel_start_time(upload.session).time())
    else:
        params['acq_datetime'] = upload.session.start
    return params # type: ignore


def is_ephys_session(session: np_session.Session) -> bool:
    return bool(next(session.npexp_path.rglob('settings*.xml'), None))

def get_np_session(session_path_or_folder_name: str) -> np_session.Session:
    """Accommodates surface channel folders, and updates the returned instance's
    npexp_path accordingly"""
    is_surface_channel_recording = utils.is_surface_channel_recording(session_path_or_folder_name)
    session = np_session.Session(session_path_or_folder_name)
    if is_surface_channel_recording and not utils.is_surface_channel_recording(session.npexp_path.name):
        # manually assign surface channel path which was lost when creating
        # session object
        session = np_session.Session(session.npexp_path.parent / f'{session.folder}_surface_channels')
        if 'surface_channels' not in session.npexp_path.name or not session.npexp_path.exists():
            raise FileNotFoundError(f"Surface channel path {session.npexp_path} does not exist, or does not exist in expected folder (ie np-exp)")
    return session

def create_codeocean_upload(
    session_path_or_folder_name: str,
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

    session = get_np_session(str(session_path_or_folder_name))
    if utils.is_surface_channel_recording(str(session_path_or_folder_name)):
        root = codeocean_root / f'{session.folder}_surface_channels'
        behavior = None
        behavior_videos = None
    else:
        root = codeocean_root / session.folder
        behavior = np_config.normalize_path(root / 'behavior')
        behavior_videos = behavior.with_name('behavior-videos')

    logger.debug(f'Created directory {root} for CodeOcean upload')

    logger.info('Attempting to create sub directory for AIND metadata jsons..')
    metadata_path = get_aind_metadata_path(root)
    
    return CodeOceanUpload(
        session = session, 
        behavior = behavior,
        behavior_videos = behavior_videos,
        ephys = np_config.normalize_path(root / 'ephys') if is_ephys_session(session) else None,
        aind_metadata = metadata_path if has_metadata(session) else None,
        job = np_config.normalize_path(root / 'upload.csv'),
        force_cloud_sync=force_cloud_sync,
        platform=platform,
    )

def has_metadata(session: np_session.Session) -> bool:
    return any(
        (session.npexp_path / f"{name}.json").exists()
        for name in utils.AIND_METADATA_NAMES
    )
    
def get_aind_metadata_path(upload_root: pathlib.Path) -> pathlib.Path:
    return np_config.normalize_path(upload_root / 'aind_metadata')

def upload_session(
    session_path_or_folder_name: str, 
    recording_dirs: Iterable[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    test: bool = False,
    hpc_upload_job_email: str = utils.HPC_UPLOAD_JOB_EMAIL,
    regenerate_symlinks: bool = True,
    adjust_ephys_timestamps: bool = True,
    codeocean_pipeline_settings: dict[str, PipelineMonitorSettings] | None = None,
    extra_UploadJobConfigsV2_params: dict[str, Any] | None = None,
) -> None:
    codeocean_root = np_session.NPEXP_PATH / ('codeocean-dev' if test else 'codeocean')
    logger.debug(f'{codeocean_root = }')
    upload = create_codeocean_upload(
        str(session_path_or_folder_name),
        codeocean_root=codeocean_root,
        recording_dirs=recording_dirs,
        force_cloud_sync=force
    )
    if regenerate_symlinks and upload.root.exists():
        logger.debug(f'Removing existing {upload.root = }')
        shutil.rmtree(upload.root.as_posix(), ignore_errors=True)
    if upload.aind_metadata:
        create_aind_metadata_symlinks(upload)
    if upload.ephys:
        create_ephys_symlinks(upload.session, upload.ephys, recording_dirs=recording_dirs)
    if upload.behavior:
        create_behavior_symlinks(upload.session, upload.behavior)
    if upload.behavior_videos:
        create_behavior_videos_symlinks(upload.session, upload.behavior_videos)
    timestamps_adjusted = False
    if adjust_ephys_timestamps and upload.ephys:
        if not upload.behavior: # includes surface channel recordings
            logger.warning(f"Cannot adjust ephys timestamps for {upload.session} - no behavior folder supplied for upload")
        else:
            try:
                utils.write_corrected_ephys_timestamps(ephys_dir=upload.ephys, behavior_dir=upload.behavior)
            except utils.SyncFileNotFoundError:
                raise FileNotFoundError(
                    (
                        f"Cannot adjust timestamps - no sync file found in {upload.behavior}. "
                        "If the session doesn't have one, run with "
                        "`adjust_ephys_timestamps=False` or `--no-sync` flag in CLI"
                    )
                ) from None
            else:
                timestamps_adjusted = True
    for path in (upload.ephys, upload.behavior, upload.behavior_videos, upload.aind_metadata):
        if path is not None and path.exists():
            utils.convert_symlinks_to_posix(path)
    job_params_from_session: dict = get_upload_params_from_session(upload)
    np_logging.web('np_codeocean').info(f'Submitting {upload.session} to hpc upload queue')
    if extra_UploadJobConfigsV2_params is None:
        extra_UploadJobConfigsV2_params = {}
    if 'codeocean_pipeline_settings' in extra_UploadJobConfigsV2_params:
        raise ValueError(
            "Cannot pass `codeocean_pipeline_settings` as a parameter to `extra_UploadJobConfigsV2_params`. "
            "Use `codeocean_pipeline_settings` parameter instead."
        )
    utils.put_jobs_for_hpc_upload(
        utils.create_upload_job_configs_v2(
            **job_params_from_session,
            codeocean_pipeline_settings=codeocean_pipeline_settings,
            check_timestamps=timestamps_adjusted,
            test=test,
            **extra_UploadJobConfigsV2_params
        ),
        upload_service_url=utils.DEV_SERVICE if test else utils.AIND_DATA_TRANSFER_SERVICE,
        user_email=hpc_upload_job_email,
        dry_run=dry_run,
        save_path=upload.job.with_suffix('.json'),
    )
    if not dry_run:
        logger.info(f'Finished submitting {upload.session} - check progress at {utils.DEV_SERVICE if test else utils.AIND_DATA_TRANSFER_SERVICE}')
    
    if (is_split_recording := 
        recording_dirs is not None 
        and len(tuple(recording_dirs)) > 1 
        and isinstance(recording_dirs, str)
    ):
        logger.warning(f"Split recording {upload.session} will need to be sorted manually with `CONCAT=True`")

if __name__ == '__main__':
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE),
    )