from __future__ import annotations

import contextlib
import csv
import datetime
import functools
import itertools
import json
import logging
import os
import pathlib
import re
from collections.abc import Generator, Iterable
from typing import Any, Literal

import np_config
import np_tools
import npc_ephys
import npc_session
import npc_sync
import numpy as np
import polars as pl
import requests
import typing_extensions
from aind_codeocean_pipeline_monitor.models import PipelineMonitorSettings
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.platforms import Platform
from aind_data_transfer_service.models.core import (
    SubmitJobRequestV2,
    Task,
    UploadJobConfigsV2,
)
from aind_slurm_rest_v2.models.v0040_job_desc_msg import (
    V0040JobDescMsg,
)

logger = logging.getLogger(__name__)

AINDPlatform = Literal["ecephys", "behavior"]

AIND_DATA_TRANSFER_SERVICE = "http://aind-data-transfer-service"
DEV_SERVICE = "http://aind-data-transfer-service-dev"
HPC_UPLOAD_JOB_EMAIL = "ben.hardcastle@alleninstitute.org"
ACQ_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

AIND_METADATA_NAMES: tuple[str, ...] = (
    "session",
    "data_description",
    "procedures",
    "processing",
    "rig",
    "subject",
)

# In the future, default slurm settings can be stored in a job_type in AWS Param Store
# see http://aind-data-transfer-service/job_params for current job_types
_DEFAULT_EPHYS_SLURM_SETTINGS_JSON = {
    "memory_per_cpu": {"set": True, "number": 8000},
    "minimum_cpus_per_node": 12,  # 6 probes * (lfp + ap)
    "partition": "aind",
    "tasks": 1,
    "time_limit": {"set": True, "number": 15 * 60},
    "environment": [
        "PATH=/bin:/usr/bin/:/usr/local/bin/",
        "LD_LIBRARY_PATH=/lib/:/lib64/:/usr/local/lib",
    ],
    "maximum_nodes": 1,
    "minimum_nodes": 1,
    "current_working_directory": ".",
}
"""Increased timelimit and cpus for running ephys compression on the hpc"""
DEFAULT_EPHYS_SLURM_SETTINGS = V0040JobDescMsg.model_validate(
    {
        **_DEFAULT_EPHYS_SLURM_SETTINGS_JSON,
        "qos": "production",
        "standard_error": "/allen/aind/scratch/svc_aind_airflow/prod/logs/%x_%j_error.out",
        "standard_output": "/allen/aind/scratch/svc_aind_airflow/prod/logs/%x_%j.out",
    }
)
DEFAULT_EPHYS_SLURM_SETTINGS_DEV = V0040JobDescMsg.model_validate(
    {
        **_DEFAULT_EPHYS_SLURM_SETTINGS_JSON,
        "qos": "dev",
        "standard_error": "/allen/aind/scratch/svc_aind_airflow/dev/logs/%x_%j_error.out",
        "standard_output": "/allen/aind/scratch/svc_aind_airflow/dev/logs/%x_%j.out",
    }
)
DEFAULT_EPHYS_IMAGE = {
    "image": "ghcr.io/allenneuraldynamics/aind-ephys-transformation",
    "image_version": "0.2.1",
    "command_script": "#!/bin/bash \nsingularity exec --cleanenv docker://%IMAGE:%IMAGE_VERSION python -m aind_ephys_transformation.ephys_job --job-settings ' %JOB_SETTINGS '",
}


class SyncFileNotFoundError(FileNotFoundError):
    pass


@functools.cache
def get_project_config() -> dict[str, Any]:
    """Config for this project"""
    return np_config.fetch("/projects/np_codeocean")


def set_npc_lims_credentials() -> None:
    creds = np_config.fetch("/projects/np_codeocean/npc_lims")
    for k, v in creds.items():
        os.environ.setdefault(k, v)


def get_home() -> pathlib.Path:
    if os.name == "nt":
        return pathlib.Path(os.environ["USERPROFILE"])
    return pathlib.Path(os.environ["HOME"])


def is_behavior_video_file(path: pathlib.Path) -> bool:
    if path.is_dir() or path.suffix not in (".mp4", ".avi", ".json"):
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
    return "surface_channels" in path_name.lower()


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
        events_dir = (
            continuous_dir.parent.parent / "events" / continuous_dir.name / "TTL"
        )
        filenames = ("continuous.dat", "timestamps.npy", "sample_numbers.npy")
        dirs = (continuous_dir,) + ((events_dir,) if events_dir.exists() else ())
        mark_for_removal = False
        for d in dirs:
            if not d.exists():
                continue
            for filename in filenames:
                if filename == "continuous.dat" and d.name == "TTL":
                    continue  # no continuous.dat expected in TTL events
                file = d / filename
                if not (file.is_symlink() or file.exists()):
                    logger.warning(
                        f"Critical file not found {file}, insufficient data for processing"
                    )
                    mark_for_removal = True
                    break
                try:
                    data = np.memmap(
                        decode_symlink_path(file),
                        dtype="int16" if "timestamps" not in file.name else "float64",
                        mode="r",
                    )
                except Exception as exc:
                    logger.warning(f"Failed to read {file}: {exc!r}")
                    mark_for_removal = True
                    break
                if data.size == 0:
                    logger.warning(f"Empty file {file}")
                    mark_for_removal = True
                    break
                logger.debug(f"Found readable, non-empty data in {file}")
            if mark_for_removal:
                break
        if mark_for_removal:
            logger.warning(f"Removing {continuous_dir} and its contents")
            remove_folder_of_symlinks(continuous_dir)
            logger.warning(f"Removing {events_dir.parent} and its contents")
            remove_folder_of_symlinks(events_dir.parent)


def remove_duplicate_ephys_data(toplevel_dir: pathlib.Path) -> None:
    logger.info("Checking for duplicate ephys data...")
    paths = sorted(ephys_continuous_dir_generator(toplevel_dir))
    experiments = set(
        re.findall(r"/experiment(\d+)/", path.as_posix())[0] for path in paths
    )
    logger.debug(f"Found {len(experiments)} experiments")
    for experiment in experiments:
        exp_paths = sorted(
            path for path in paths if f"experiment{experiment}" in path.as_posix()
        )
        recordings = set(
            re.findall(r"/recording(\d+)/", path.as_posix())[0] for path in exp_paths
        )
        logger.debug(f"Found {len(recordings)} recordings in experiment{experiment}")
        for recording in recordings:
            recording_paths = sorted(
                path for path in exp_paths if f"recording{recording}" in path.as_posix()
            )
            probes = []
            # import pdb; pdb.set_trace()
            for continuous_dir in recording_paths:
                try:
                    probe = npc_session.ProbeRecord(continuous_dir.name)
                except ValueError:
                    continue
                suffix = continuous_dir.name.split("-")[-1]
                assert suffix in ("AP", "LFP")
                recording_name = f"{probe}-{suffix}"
                if recording_name in probes:
                    logger.info(
                        f"Duplicate {recording_name = } found in {continuous_dir.parent.parent} - removing"
                    )
                    remove_folder_of_symlinks(continuous_dir)
                else:
                    probes.append(recording_name)


def remove_folder_of_symlinks(folder: pathlib.Path) -> None:
    """Recursive deletion of all files in dir tree, with a check that each is a
    symlink."""
    for path in folder.rglob("*"):
        if path.is_dir():
            remove_folder_of_symlinks(path)
        else:
            assert path.is_symlink(), f"Expected {path} to be a symlink"
            path.unlink(missing_ok=True)
    with contextlib.suppress(FileNotFoundError):
        folder.rmdir()


def ephys_recording_dir_generator(
    toplevel_dir: pathlib.Path,
) -> Generator[pathlib.Path, None, None]:
    for recording_dir in toplevel_dir.rglob("recording[0-9]*"):
        if recording_dir.is_dir():
            yield recording_dir


def ephys_continuous_dir_generator(
    toplevel_dir: pathlib.Path,
) -> Generator[pathlib.Path, None, None]:
    for recording_dir in ephys_recording_dir_generator(toplevel_dir):
        parent = recording_dir / "continuous"
        if not parent.exists():
            continue
        for continuous_dir in parent.iterdir():
            if continuous_dir.is_dir():
                yield continuous_dir


def ephys_structure_oebin_generator(
    toplevel_dir: pathlib.Path,
) -> Generator[pathlib.Path, None, None]:
    for recording_dir in ephys_recording_dir_generator(toplevel_dir):
        oebin_path = recording_dir / "structure.oebin"
        if not (oebin_path.is_symlink() or oebin_path.exists()):
            # symlinks that are created for the hpc use posix paths, and aren't
            # readable on windows, so .exists() returns False: use .is_symlink() instead
            logger.warning(f"No structure.oebin found in {recording_dir}")
            continue
        yield oebin_path


def cleanup_ephys_metadata(toplevel_dir: pathlib.Path) -> None:
    logger.debug("Checking structure.oebin for missing folders...")
    for oebin_path in ephys_structure_oebin_generator(toplevel_dir):
        oebin_obj = np_tools.read_oebin(decode_symlink_path(oebin_path))
        logger.debug(f"Checking {oebin_path} against actual folders...")
        any_removed = False
        for subdir_name in ("events", "continuous"):
            subdir = oebin_path.parent / subdir_name
            # iterate over copy of list so as to not disrupt iteration when elements are removed
            for device in [device for device in oebin_obj[subdir_name]]:
                if not (subdir / device["folder_name"]).exists():
                    logger.info(
                        f'{device["folder_name"]} not found in {subdir}, removing from structure.oebin'
                    )
                    oebin_obj[subdir_name].remove(device)
                    any_removed = True
        if any_removed:
            oebin_path.unlink()
            oebin_path.write_text(json.dumps(oebin_obj, indent=4))
            logger.debug(
                "Overwrote symlink to structure.oebin with corrected structure.oebin"
            )


def write_corrected_ephys_timestamps(
    ephys_dir: pathlib.Path,
    behavior_dir: pathlib.Path,
) -> None:
    for path in itertools.chain(behavior_dir.glob("*.h5"), behavior_dir.glob("*.sync")):
        with contextlib.suppress(Exception):
            sync_dataset = npc_sync.SyncDataset(path)
            _ = sync_dataset.line_labels
            logger.info(f"Found valid sync file {path.as_posix()}")
            break
    else:
        raise SyncFileNotFoundError(
            f"No valid sync file found in {behavior_dir.as_posix()}"
        )

    timing_on_pxi = (
        timing
        for timing in npc_ephys.get_ephys_timing_on_pxi(
            ephys_dir.glob("**/experiment*/recording*"),
        )
    )
    timing_on_sync = npc_ephys.get_ephys_timing_on_sync(
        sync=sync_dataset,
        devices=timing_on_pxi,
    )
    npc_ephys.overwrite_timestamps(timing_on_sync)
    logger.info(f"Corrected timestamps in {ephys_dir}")


def decode_symlink_path(oebin_path: pathlib.Path) -> pathlib.Path:
    if not oebin_path.is_symlink():
        return oebin_path
    return np_config.normalize_path(oebin_path.readlink())


def is_csv_in_hpc_upload_queue(
    csv_path: pathlib.Path, upload_service_url: str = AIND_DATA_TRANSFER_SERVICE
) -> bool:
    """Check if an upload job has been submitted to the hpc upload queue.

    - currently assumes one job per csv
    - does not check status (job may be FINISHED rather than RUNNING)

    >>> is_csv_in_hpc_upload_queue("//allen/programs/mindscope/workgroups/np-exp/codeocean/DRpilot_664851_20231114/upload.csv")
    False
    """
    # get subject-id, acq-datetime from csv
    df = pl.read_csv(csv_path, eol_char="\r")
    for col in df.get_columns():
        if col.name.startswith("subject") and col.name.endswith("id"):
            subject = npc_session.SubjectRecord(col[0])
            continue
        if col.name.startswith("acq") and "datetime" in col.name.lower():
            dt = npc_session.DatetimeRecord(col[0])
            continue
        if col.name == "platform":
            platform = col[0]
            continue
    return is_session_in_hpc_queue(
        subject=subject,
        acq_datetime=dt.dt,
        platform=platform,
        upload_service_url=upload_service_url,
    )


def is_session_in_hpc_queue(
    subject: int | str,
    acq_datetime: str | datetime.datetime,
    platform: str | None = None,
    upload_service_url: str = AIND_DATA_TRANSFER_SERVICE,
) -> bool:
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


def is_job_in_hpc_upload_queue(
    job: UploadJobConfigsV2, upload_service_url: str = AIND_DATA_TRANSFER_SERVICE
) -> bool:
    return is_session_in_hpc_queue(
        job.subject_id, job.acq_datetime, job.platform.abbreviation, upload_service_url
    )


def write_upload_csv(
    content: dict[str, Any],
    output_path: pathlib.Path,
) -> pathlib.Path:
    logger.info(f"Creating upload job file {output_path}")
    with open(output_path, "w") as f:
        w = csv.writer(f, lineterminator="")
        w.writerow(content.keys())
        w.writerow("\n")
        w.writerow(content.values())
    return output_path


def create_upload_job_configs_v2(
    project_name: str,
    platform: str,
    subject_id: str,
    force_cloud_sync: bool,
    modalities: dict[str, str],
    acq_datetime: datetime.datetime,
    user_email: str = HPC_UPLOAD_JOB_EMAIL,
    job_type: str = "default",
    metadata_dir: str | None = None,
    codeocean_pipeline_settings: dict[str, PipelineMonitorSettings] | None = None,
    check_timestamps: bool = True,  # default in transfer service is True: checks timestamps have been corrected via flag file
    test: bool = False,
    **extra_UploadJobConfigsV2_params: Any,
) -> UploadJobConfigsV2:
    """Create a UploadJobConfigsV2 model. Modalities should be provided in format
    {modality_abbr: input_source}. job_type refers to the default or custom
    presets used for compression and Code Ocean pipelines.
    """
    # Each task in airflow can be configured individually
    # force_cloud_sync
    check_s3_folder_exists_task = Task(skip_task=True) if force_cloud_sync else None
    # metadata_dir
    gather_preliminary_metadata_task = (
        Task(job_settings={"metadata_dir": metadata_dir})
        if metadata_dir is not None
        else None
    )
    # modality transformation settings
    modality_transformation_settings_tasks = dict()  # {modality_abbr: Task}
    if "modalities" in extra_UploadJobConfigsV2_params:
        raise ValueError(
            "modalities should not be passed as a parameter in extra_BasicUploadJobConfigs_params"
        )
    for modality_abbr, input_source in modalities.items():
        job_settings: dict[str, Any] = {
            "input_source": input_source,
            "output_directory": "%OUTPUT_LOCATION",
        }
        # Ecephys compression settings are currently hardcoded
        # In the future, these can be stored in AWS param store as part of a "job_type"
        if modality_abbr == Modality.ECEPHYS.abbreviation:
            if not check_timestamps:
                job_settings["check_timestamps"] = False
            image_resources = (
                DEFAULT_EPHYS_SLURM_SETTINGS_DEV
                if test
                else DEFAULT_EPHYS_SLURM_SETTINGS
            ).model_dump(mode="json", exclude_none=True)
            modality_task = Task(
                skip_task=False,
                job_settings=job_settings,
                image_resources=image_resources,
                **DEFAULT_EPHYS_IMAGE,
            )
        else:
            modality_task = Task(
                job_settings=job_settings,
            )
        modality_transformation_settings_tasks[modality_abbr] = modality_task

    # Code Ocean pipeline settings
    # You can manually specify up to one pipeline conf per modality.
    # These will override any pipelines defined by the job_type.
    # In the future, these can be stored in AWS param store as part of a "job_type"
    codeocean_pipeline_settings_tasks = dict()  # {modality_abbr: Task}
    if codeocean_pipeline_settings is not None:
        codeocean_pipeline_settings_tasks = {
            k: Task(
                job_settings={
                    "pipeline_monitor_settings": v.model_dump(
                        mode="json", exclude_none=True
                    )
                }
            )
            for k, v in codeocean_pipeline_settings.items()
        }
    tasks = {
        "check_s3_folder_exists": check_s3_folder_exists_task,
        "gather_preliminary_metadata": gather_preliminary_metadata_task,
        "modality_transformation_settings": modality_transformation_settings_tasks,
        "codeocean_pipeline_settings": codeocean_pipeline_settings_tasks,
    }
    return UploadJobConfigsV2(
        job_type=job_type,
        platform=Platform.from_abbreviation(platform),
        project_name=project_name,
        subject_id=subject_id,
        acq_datetime=acq_datetime,
        modalities=[
            Modality.from_abbreviation(m)
            for m in modality_transformation_settings_tasks.keys()
        ],
        tasks={k: v for k, v in tasks.items() if v is not None},
        user_email=user_email,
        **extra_UploadJobConfigsV2_params,
    )


def put_jobs_for_hpc_upload(
    upload_jobs: UploadJobConfigsV2 | Iterable[UploadJobConfigsV2],
    upload_service_url: str = AIND_DATA_TRANSFER_SERVICE,
    user_email: str = HPC_UPLOAD_JOB_EMAIL,
    email_notification_types: Iterable[
        Literal["begin", "end", "fail", "retry", "all"]
    ] = ("fail",),
    dry_run: bool = False,
    save_path: pathlib.Path | None = None,
    **extra_model_kwargs: Any,
) -> None:
    """Submit one or more jobs to the aind-data-transfer-service, for
    upload to S3 on the hpc.

    - accepts one or more aind_data_transfer_service UploadJobConfigsV2 models
    - assembles a SubmitJobRequestV2 model
    - excludes jobs for sessions that are already in the upload queue
    - accepts additional parameters for SubmitJobRequestV2 as kwargs
    - submits json via http request
    - optionally saves the json file as a record
    """
    if isinstance(upload_jobs, UploadJobConfigsV2):
        upload_jobs = (upload_jobs,)
    submit_request = SubmitJobRequestV2(
        upload_jobs=[job for job in upload_jobs if not is_job_in_hpc_upload_queue(job)],
        user_email=user_email,
        email_notification_types=email_notification_types,
        **extra_model_kwargs,
    )
    post_request_content = submit_request.model_dump(mode="json", exclude_none=True)
    if save_path:
        save_path.write_text(
            submit_request.model_dump_json(
                round_trip=True, exclude_none=True, indent=4
            ),
            errors="ignore",
        )
    if dry_run:
        logger.warning(
            f"Dry run: not submitting {len(upload_jobs)} upload job(s) to {upload_service_url}"
        )
        return

    # Uncomment to perform a validation check:
    # validate_json_response: requests.Response = requests.post(
    #     url=f"{upload_service_url}/api/v2/validate_json",
    #     json=post_request_content,
    # )
    # validate_json_response.raise_for_status()

    # Submit the jobs
    post_json_response: requests.Response = requests.post(
        url=f"{upload_service_url}/api/v2/submit_jobs",
        json=post_request_content,
    )
    logger.info(f"Submitted {len(upload_jobs)} upload job(s) to {upload_service_url}")
    post_json_response.raise_for_status()


@typing_extensions.deprecated(
    "Uses old, pre-v1 endpoints: use put_jobs_for_hpc_upload in combination with create_upload_job_configs_v2"
)
def put_csv_for_hpc_upload(
    csv_path: pathlib.Path,
    upload_service_url: str = AIND_DATA_TRANSFER_SERVICE,
    hpc_upload_job_email: str = HPC_UPLOAD_JOB_EMAIL,
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
                response.json()["data"]["errors"]
            except (
                KeyError,
                IndexError,
                requests.exceptions.JSONDecodeError,
                SyntaxError,
            ) as exc1:
                try:
                    response.raise_for_status()
                except requests.exceptions.HTTPError as exc2:
                    raise exc2 from exc1

    with open(csv_path, "rb") as f:
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
        logger.info(
            f"Dry run: not submitting {csv_path} to hpc upload queue at {upload_service_url}."
        )
        return
    post_csv_response = requests.post(
        url=f"{upload_service_url}/api/submit_hpc_jobs",
        json=dict(
            jobs=[
                dict(
                    hpc_settings=json.dumps(
                        {"time_limit": 60 * 15, "mail_user": hpc_upload_job_email}
                    ),
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
    if posix.startswith("//"):
        posix = posix[1:]
    return posix


def convert_symlinks_to_posix(toplevel_dir: str | pathlib.Path) -> None:
    """Convert all symlinks in `root_dir` (recursively) to POSIX paths. This is a
    necessary last step before submitting uploads to run on the HPC.
    """
    for path in pathlib.Path(toplevel_dir).rglob("*"):
        if path.is_symlink():
            posix_target = path.readlink().as_posix().removeprefix("//?/UNC")
            path.unlink()
            np_tools.symlink(src=ensure_posix(posix_target), dest=path)


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=doctest.ELLIPSIS
        | doctest.NORMALIZE_WHITESPACE
        | doctest.IGNORE_EXCEPTION_DETAIL
    )
