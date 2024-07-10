from __future__ import annotations

import contextlib
import csv
import datetime
import functools
import json
import logging
import os
import pathlib
from typing import Any, Generator, Iterable, Literal
import typing_extensions

import aind_data_transfer_models.core
import aind_slurm_rest.models
import np_config
import np_tools
import npc_session
import numpy as np
import polars as pl
import requests

logger = logging.getLogger(__name__)

AINDPlatform = Literal['ecephys', 'behavior']

AIND_DATA_TRANSFER_SERVICE = "http://aind-data-transfer-service"
DEV_SERVICE = "http://aind-data-transfer-service-dev"
HPC_UPLOAD_JOB_EMAIL = "ben.hardcastle@alleninstitute.org"
ACQ_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

AIND_METADATA_NAMES: tuple[str, ...] = ('session', 'data_description', 'procedures', 'processing', 'rig', 'subject')

DEFAULT_EPHYS_SLURM_SETTINGS = aind_slurm_rest.models.V0036JobProperties(
    environment=dict(),  # JonY: set this to an empty dictionary
    time_limit = 15 * 60,
    minimum_cpus_per_node=12, # 6 probes * (lfp + ap)
)
"""Increased timelimit and cpus for running ephys compression on the hpc"""

@functools.cache
def get_project_config() -> dict[str, Any]:
    """Config for this project"""
    return np_config.fetch('/projects/np_codeocean')

def set_npc_lims_credentials() -> None:
    creds = np_config.fetch('/projects/np_codeocean/npc_lims')
    for k, v in creds.items():
        os.environ.setdefault(k, v)
        
def get_home() -> pathlib.Path:
    if os.name == 'nt':
        return pathlib.Path(os.environ['USERPROFILE'])
    return pathlib.Path(os.environ['HOME'])

def is_behavior_video_file(path: pathlib.Path) -> bool:
    if path.is_dir() or path.suffix not in ('.mp4', '.avi', '.json'):
        return False
    with contextlib.suppress(ValueError):
        _ = npc_session.extract_mvr_camera_name(path.as_posix())
        return True
    return False
    
def is_surface_channel_recording(path_name: str) -> bool:
    """
    >>> import np_session
    >>> session = np_session.Session("//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_690706_20231129_surface_channels")
    >>> is_surface_channel_recording(session.npexp_path.as_posix())
    True
    """
    return 'surface_channels' in path_name.lower()

def cleanup_ephys_symlinks(toplevel_dir: pathlib.Path) -> None:
    """After creating symlinks to the ephys data, run this to make any necessary 
    modifications prior to upload.
    
    Provided dir path should be a directory containing all ephys data in
    subfolders (e.g. directory containing "Record Node 10x" folders)
    
    Only deletes symlinks or writes new files in place of symlinks - does not
    modify original data.
    
    Rules:
    - if any continuous.dat files are unreadable: remove them and their containing folders
    - if any probes were recorded on multiple record nodes: just keep the first
    - if continuous.dat files are missing (ie. excluded because probes weren't
      inserted, or we removed symlinks in previous steps): update metadata files
    """
    remove_unreadable_ephys_data(toplevel_dir)
    remove_duplicate_ephys_data(toplevel_dir)
    cleanup_ephys_metadata(toplevel_dir)

def remove_unreadable_ephys_data(toplevel_dir: pathlib.Path) -> None:
    
    for continuous_dir in ephys_continuous_dir_generator(toplevel_dir):
        events_dir = continuous_dir.parent.parent / 'events' / continuous_dir.name / 'TTL'
        filenames = ('continuous.dat', 'timestamps.npy', 'sample_numbers.npy')    
        dirs = (continuous_dir, ) + ((events_dir,) if events_dir.exists() else ())
        mark_for_removal = False
        for d in dirs:
            if not d.exists():
                continue
            for filename in filenames:
                if filename == 'continuous.dat' and d.name == 'TTL':
                    continue # no continuous.dat expected in TTL events
                file = d / filename
                if not (file.is_symlink() or file.exists()):
                    logger.warning(f'Critical file not found {file}, insufficient data for processing')
                    mark_for_removal = True
                    break
                try:
                    data = np.memmap(decode_symlink_path(file), dtype="int16" if 'timestamps' not in file.name else "float64", mode="r")
                except Exception as exc:
                    logger.warning(f'Failed to read {file}: {exc!r}')
                    mark_for_removal = True
                    break
                if data.size == 0:
                    logger.warning(f'Empty file {file}')
                    mark_for_removal = True
                    break
                logger.debug(f'Found readable, non-empty data in {file}')
            if mark_for_removal:
                break
        if mark_for_removal:
            logger.warning(f'Removing {continuous_dir} and its contents')
            remove_folder_of_symlinks(continuous_dir)
            logger.warning(f'Removing {events_dir.parent} and its contents')
            remove_folder_of_symlinks(events_dir.parent)
            
def remove_duplicate_ephys_data(toplevel_dir: pathlib.Path) -> None:
    previous_recording_name = ''
    for continuous_dir in ephys_continuous_dir_generator(toplevel_dir):
        recording_name = continuous_dir.parent.parent.name
        if recording_name != previous_recording_name:
            # reset probes list for each new recording
            probes = []
        try:
            probe = npc_session.ProbeRecord(continuous_dir.name)
        except ValueError:
            continue
        suffix = continuous_dir.name.split('-')[-1]
        assert suffix in ('AP', 'LFP')
        recording_name = f"{probe}-{suffix}"
        if recording_name in probes:
            logger.info(f'Duplicate {recording_name = } found in {continuous_dir.parent.parent} - removing')
            remove_folder_of_symlinks(continuous_dir)
        else:
            probes.append(recording_name)
            
def remove_folder_of_symlinks(folder: pathlib.Path) -> None:
    """Recursive deletion of all files in dir tree, with a check that each is a
    symlink."""
    for path in folder.rglob('*'):
        if path.is_dir():
            remove_folder_of_symlinks(path)
        else:
            assert path.is_symlink(), f'Expected {path} to be a symlink'
            path.unlink(missing_ok=True)
    with contextlib.suppress(FileNotFoundError):
        folder.rmdir()

def ephys_recording_dir_generator(toplevel_dir: pathlib.Path) -> Generator[pathlib.Path, None, None]:
    for recording_dir in toplevel_dir.rglob('recording[0-9]*'):
        if recording_dir.is_dir():
            yield recording_dir
            
def ephys_continuous_dir_generator(toplevel_dir: pathlib.Path) -> Generator[pathlib.Path, None, None]:
    for recording_dir in ephys_recording_dir_generator(toplevel_dir):
        parent = recording_dir / 'continuous'
        if not parent.exists():
            continue
        for continuous_dir in parent.iterdir():
            if continuous_dir.is_dir():
                yield continuous_dir

def ephys_structure_oebin_generator(toplevel_dir: pathlib.Path) -> Generator[pathlib.Path, None, None]:
    for recording_dir in ephys_recording_dir_generator(toplevel_dir):
        oebin_path = recording_dir / 'structure.oebin'
        if not (oebin_path.is_symlink() or oebin_path.exists()): 
            # symlinks that are created for the hpc use posix paths, and aren't
            # readable on windows, so .exists() returns False: use .is_symlink() instead
            logger.warning(f'No structure.oebin found in {recording_dir}')
            continue
        yield oebin_path
        
def cleanup_ephys_metadata(toplevel_dir: pathlib.Path) -> None:
    logger.debug('Checking structure.oebin for missing folders...')
    for oebin_path in ephys_structure_oebin_generator(toplevel_dir):
        oebin_obj = np_tools.read_oebin(decode_symlink_path(oebin_path))
        logger.debug(f'Checking {oebin_path} against actual folders...')
        any_removed = False
        for subdir_name in ('events', 'continuous'):    
            subdir = oebin_path.parent / subdir_name
            # iterate over copy of list so as to not disrupt iteration when elements are removed
            for device in [device for device in oebin_obj[subdir_name]]:
                if not (subdir / device['folder_name']).exists():
                    logger.info(f'{device["folder_name"]} not found in {subdir}, removing from structure.oebin')
                    oebin_obj[subdir_name].remove(device)
                    any_removed = True
        if any_removed:
            oebin_path.unlink()
            oebin_path.write_text(json.dumps(oebin_obj, indent=4))
            logger.debug('Overwrote symlink to structure.oebin with corrected structure.oebin')

def decode_symlink_path(oebin_path: pathlib.Path) -> pathlib.Path:
    if not oebin_path.is_symlink():
        return oebin_path
    return np_config.normalize_path(oebin_path.readlink())

def is_csv_in_hpc_upload_queue(csv_path: pathlib.Path, upload_service_url: str = AIND_DATA_TRANSFER_SERVICE) -> bool:
    """Check if an upload job has been submitted to the hpc upload queue.

    - currently assumes one job per csv
    - does not check status (job may be FINISHED rather than RUNNING)

    >>> is_csv_in_hpc_upload_queue("//allen/programs/mindscope/workgroups/np-exp/codeocean/DRpilot_664851_20231114/upload.csv")
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
        if col.name == 'platform':
            platform = col[0]
            continue
    return is_session_in_hpc_queue(subject=subject, acq_datetime=dt.dt, platform=platform, upload_service_url=upload_service_url)

def is_session_in_hpc_queue(subject: int | str, acq_datetime: str | datetime.datetime, platform: str | None = None, upload_service_url: str = AIND_DATA_TRANSFER_SERVICE) -> bool:
    """
    >>> is_session_in_hpc_queue(366122, datetime.datetime(2023, 11, 14, 0, 0, 0))
    False
    >>> is_session_in_hpc_queue(702136, datetime.datetime(2024, 3, 4, 13, 21, 35))
    True
    """
    if not isinstance(acq_datetime, datetime.datetime):
        acq_datetime = datetime.datetime.strptime(acq_datetime, ACQ_DATETIME_FORMAT)
    partial_session_id = f"{subject}_{acq_datetime.strftime(ACQ_DATETIME_FORMAT).replace(' ', '_').replace(':', '-')}"
    if platform:
        partial_session_id = f"{platform}_{partial_session_id}"
        
    jobs_response = requests.get(f"{upload_service_url}/jobs")
    jobs_response.raise_for_status()
    return partial_session_id in jobs_response.content.decode()

def is_job_in_hpc_upload_queue(job: aind_data_transfer_models.core.BasicUploadJobConfigs, upload_service_url: str = AIND_DATA_TRANSFER_SERVICE) -> bool:
    return is_session_in_hpc_queue(job.subject_id, job.acq_datetime, job.platform, upload_service_url)

def write_upload_csv(
    content: dict[str, Any],
    output_path: pathlib.Path,
) -> pathlib.Path:
    logger.info(f'Creating upload job file {output_path}')
    with open(output_path, 'w') as f:
        w = csv.writer(f, lineterminator='')
        w.writerow(content.keys())
        w.writerow('\n')
        w.writerow(content.values())
    return output_path

def get_job_models_from_csv(
    path: pathlib.Path,
    ephys_slurm_settings: aind_slurm_rest.models.V0036JobProperties = DEFAULT_EPHYS_SLURM_SETTINGS,
    user_email: str = HPC_UPLOAD_JOB_EMAIL,
) -> tuple[aind_data_transfer_models.core.BasicUploadJobConfigs, ...]:
    jobs = pl.read_csv(path, eol_char='\r').with_columns(
        pl.col('subject-id').cast(str),
    ).to_dicts()
    jobs = jobs
    models = []
    for job in jobs.copy():
        modalities = []
        for modality_column in (k for k in job.keys() if k.startswith('modality') and ".source" not in k):
            modality_name = job[modality_column]
            modalities.append(
                aind_data_transfer_models.core.ModalityConfigs(
                    modality=modality_name,
                    source=job[f"{modality_column}.source"],
                    slurm_settings = ephys_slurm_settings if modality_name == 'ecephys' else None,
                    ),
                )
        for k in (k for k in job.copy().keys() if k.startswith('modality')):
            del job[k]
        for k, v in job.items():
            if isinstance(v, str) and '\n' in v:
                job[k] = v.replace('\n', '')
        models.append(
            aind_data_transfer_models.core.BasicUploadJobConfigs(
                **{k.replace('-', '_'): v for k,v in job.items()},
                modalities=modalities,
                user_email=user_email,
            )
        )
    return tuple(models)

def put_jobs_for_hpc_upload(
    upload_jobs: aind_data_transfer_models.core.BasicUploadJobConfigs | Iterable[aind_data_transfer_models.core.BasicUploadJobConfigs],
    upload_service_url: str = AIND_DATA_TRANSFER_SERVICE,
    user_email: str = HPC_UPLOAD_JOB_EMAIL,
    email_notification_types: Iterable[str | aind_data_transfer_models.core.EmailNotificationType] = ('fail',),
    dry_run: bool = False,
    save_path: pathlib.Path | None = None,
    **extra_model_kwargs: Any,
) -> None:
    """Submit one or more jobs to the aind-data-transfer-service, for
    upload to S3 on the hpc.
    
    - accepts one or more aind_data_schema BasicUploadJobConfigs models
    - assembles a SubmitJobRequest model
    - excludes jobs for sessions that are already in the upload queue
    - accepts additional parameters for SubmitHpcJobRequest as kwargs
    - submits json via http request
    - optionally saves the json file as a record
    """
    if not isinstance(upload_jobs, Iterable):
        upload_jobs = (upload_jobs, )
    submit_request = aind_data_transfer_models.core.SubmitJobRequest(
        upload_jobs=[job for job in upload_jobs if not is_job_in_hpc_upload_queue(job)],
        user_email=user_email,
        email_notification_types=email_notification_types,
        **extra_model_kwargs,
    )
    post_request_content = json.loads(
        submit_request.model_dump_json(round_trip=True, exclude_none=True)
    ) #! round_trip required for s3 bucket suffix to work correctly
    if save_path:
        save_path.write_text(submit_request.model_dump_json(round_trip=True, indent=4), errors='ignore')
    if dry_run:
        logger.warning(f'Dry run: not submitting {len(upload_jobs)} upload job(s) to {upload_service_url}')
        return
    post_json_response: requests.Response = requests.post(
        url=f"{upload_service_url}/api/v1/submit_jobs",
        json=post_request_content,
    )
    logger.info(f"Submitted {len(upload_jobs)} upload job(s) to {upload_service_url}")
    post_json_response.raise_for_status()

@typing_extensions.deprecated("Uses old, pre-v1 endpoints: use put_jobs_for_hpc_upload in combination with get_job_models_from_csv")
def put_csv_for_hpc_upload(
    csv_path: pathlib.Path,
    upload_service_url: str = AIND_DATA_TRANSFER_SERVICE,
    hpc_upload_job_email: str =  HPC_UPLOAD_JOB_EMAIL,
    dry_run: bool = False,
) -> None:
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
                response.json()['data']['errors']
            except (KeyError, IndexError, requests.exceptions.JSONDecodeError, SyntaxError) as exc1:
                try:
                    response.raise_for_status()
                except requests.exceptions.HTTPError as exc2:
                    raise exc2 from exc1
                
    with open(csv_path, 'rb') as f:
        validate_csv_response = requests.post(
            url=f"{upload_service_url}/api/validate_csv", 
            files=dict(file=f),
            )
    _raise_for_status(validate_csv_response)
    logger.debug(f"Validated response: {validate_csv_response.json()}")
    if is_csv_in_hpc_upload_queue(csv_path, upload_service_url):
        logger.warning(f"Job already submitted for {csv_path}")
        return
    if dry_run:
        logger.info(f'Dry run: not submitting {csv_path} to hpc upload queue at {upload_service_url}.')
        return
    post_csv_response = requests.post(
        url=f"{upload_service_url}/api/submit_hpc_jobs", 
        json=dict(
            jobs=[
                    dict(
                        hpc_settings=json.dumps({"time_limit": 60 * 15, "mail_user": hpc_upload_job_email}),
                        upload_job_settings=validate_csv_response.json()["data"]["jobs"][0],
                        script="",
                    )
                ]
        ),
    )
    logger.info(f"Submitted {csv_path} to hpc upload queue at {upload_service_url}")
    _raise_for_status(post_csv_response)


def ensure_posix(path: str | pathlib.Path) -> str:
    posix = pathlib.Path(path).as_posix()
    if posix.startswith('//'):
        posix = posix[1:]
    return posix


if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE | doctest.IGNORE_EXCEPTION_DETAIL)