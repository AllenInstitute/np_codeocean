from __future__ import annotations

import configparser
import contextlib
import csv
import functools
import json
import logging
import os
import pathlib
from typing import Any, Literal

import np_config
import npc_session
import np_tools
import polars as pl
import requests

logger = logging.getLogger(__name__)

AINDPlatform = Literal['ecephys', 'behavior']

AIND_DATA_TRANSFER_SERVICE = "http://aind-data-transfer-service"
DEV_SERVICE = "http://aind-data-transfer-service-dev"
HPC_UPLOAD_JOB_EMAIL = "arjun.sridhar@alleninstitute.org"
ACQ_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

@functools.cache
def get_project_config() -> dict[str, Any]:
    """Config for this project"""
    return np_config.fetch('/projects/np_codeocean')

@functools.cache
def get_aws_config() -> dict[Literal['aws_access_key_id', 'aws_secret_access_key'], str]:
    """Config for connecting to AWS/S3 via awscli/boto3"""
    return np_config.fetch('/projects/np_codeocean/aws')['config']

@functools.cache
def get_aws_credentials() -> dict[Literal['domain', 'token'], str]:
    """Config for connecting to AWS/S3 via awscli/boto3"""
    return np_config.fetch('/projects/np_codeocean/aws')['credentials']

@functools.cache
def get_codeocean_config() -> dict[Literal['region'], str]:
    """Config for connecting to CodeOcean via http API"""
    return np_config.fetch('/projects/np_codeocean/codeocean')['credentials']

def get_home() -> pathlib.Path:
    if os.name == 'nt':
        return pathlib.Path(os.environ['USERPROFILE'])
    return pathlib.Path(os.environ['HOME'])

def get_aws_files() -> dict[Literal['config', 'credentials'], pathlib.Path]:
    return {
        'config': get_home() / '.aws' / 'config',
        'credentials': get_home() / '.aws' / 'credentials',
    }

def get_codeocean_files() -> dict[Literal['credentials'], pathlib.Path]:
    return {
        'credentials': get_home() / '.codeocean' / 'credentials.json',
    }

def verify_ini_config(path: pathlib.Path, contents: dict, profile: str = 'default') -> None:
    config = configparser.ConfigParser()
    if path.exists():
        config.read(path)
    if not all(k in config[profile] for k in contents):
        raise ValueError(f'Profile {profile} in {path} exists but is missing some keys required for codeocean or s3 access.')
    
def write_or_verify_ini_config(path: pathlib.Path, contents: dict, profile: str = 'default') -> None:
    config = configparser.ConfigParser()
    if path.exists():
        config.read(path)
        try:    
            verify_ini_config(path, contents, profile)
        except ValueError:
            pass
        else:   
            return
    config[profile] = contents
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with path.open('w') as f:
        config.write(f)
    verify_ini_config(path, contents, profile)

def verify_json_config(path: pathlib.Path, contents: dict) -> None:
    config = json.loads(path.read_text())
    if not all(k in config for k in contents):
        raise ValueError(f'{path} exists but is missing some keys required for codeocean or s3 access.')
    
def write_or_verify_json_config(path: pathlib.Path, contents: dict) -> None:
    if path.exists():
        try:
            verify_json_config(path, contents)
        except ValueError:
            contents = np_config.merge(json.loads(path.read_text()), contents)
        else:   
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    path.write_text(json.dumps(contents, indent=4))
    
def ensure_credentials() -> None:
    for file, contents in (
        (get_aws_files()['config'], get_aws_config()),
        (get_aws_files()['credentials'], get_aws_credentials()),
    ):
        write_or_verify_ini_config(file, contents, profile='default')
    
    for file, contents in (
        (get_codeocean_files()['credentials'], get_codeocean_config()),
    ):
        write_or_verify_json_config(file, contents)


def is_behavior_video_file(path: pathlib.Path) -> bool:
    if path.is_dir() or path.suffix not in ('.mp4', '.avi', '.json'):
        return False
    with contextlib.suppress(ValueError):
        _ = npc_session.extract_mvr_camera_name(path.as_posix())
        return True
    return False
    
def is_surface_channel_recording(path_name: str) -> bool:
    """
    >>> session = np_session.Session("//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_690706_20231129_surface_channels")
    >>> is_surface_channel_recording(session.npexp_path.as_posix())
    True
    """
    return 'surface_channels' in path_name.lower()

def cleanup_ephys_directories(dest: pathlib.Path) -> None:
    """
    In case some probes are missing, remove device entries from structure.oebin
    files for devices with folders that have not been preserved.
    """
    logger.debug('Checking structure.oebin for missing folders...')
    recording_dirs = dest.rglob('recording[0-9]*')
    for recording_dir in recording_dirs:
        if not recording_dir.is_dir():
            continue
        oebin_path = recording_dir / 'structure.oebin'
        if not (oebin_path.is_symlink() or oebin_path.exists()):
            logger.warning(f'No structure.oebin found in {recording_dir}')
            continue
        logger.debug(f'Examining oebin: {oebin_path} for correction')
        oebin_obj = np_tools.read_oebin(np_config.normalize_path(oebin_path.readlink()))
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
            logger.debug('Overwrote symlink to structure.oebin with corrected strcuture.oebin')

def is_in_hpc_upload_queue(csv_path: pathlib.Path, upload_service_url: str = AIND_DATA_TRANSFER_SERVICE) -> bool:
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
    
    jobs_response = requests.get(f"{upload_service_url}/jobs")
    jobs_response.raise_for_status()
    return partial_session_id in jobs_response.content.decode()

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
                import pdb; pdb.set_trace()
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
    if is_in_hpc_upload_queue(csv_path, upload_service_url):
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
    _raise_for_status(post_csv_response)


def ensure_posix(path: pathlib.Path) -> str:
    posix = path.as_posix()
    if posix.startswith('//'):
        posix = posix[1:]
    return posix


if __name__ == '__main__':
    ensure_credentials()