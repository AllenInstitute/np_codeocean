import logging
import typing
from pathlib import Path

import np_config
import np_session
import datetime

from np_aind_metadata.integrations import dynamic_routing_task

from np_codeocean import upload as np_codeocean_upload


logger = logging.getLogger(__name__)


def add_metadata(
    session_directory: Path,
    session_datetime: datetime.datetime,
    modality: np_codeocean_upload.Modality,
    rig_storage_directory: Path,
) -> None:
    """Adds metadata to a session directory.

    TODO: Return created paths rather than None to better support a monadic
     pattern.
    """
    normalized = np_config.normalize_path(session_directory)
    logger.debug("Normalized session directory: %s" % normalized)
    if modality in ('ecephys', ):
        dynamic_routing_task.add_np_rig_to_session_dir(
            normalized,
            session_datetime,
            rig_storage_directory,
        )
    elif modality in ('behavior', ):
        task_paths = list(
            normalized.glob("Dynamic*.hdf5")
        )
        logger.debug("Scraped task_paths: %s" % task_paths)
        task_path = task_paths[0]
        logger.debug("Using task path: %s" % task_path)
        rig_model_path = dynamic_routing_task.copy_task_rig(
            task_path,
            normalized / "rig.json",
            rig_storage_directory,
        )
        if not rig_model_path:
            raise Exception("Failed to copy task rig.")
        logger.debug("Rig model path: %s" % rig_model_path)
        session_model_path = dynamic_routing_task.scrape_session_model_path(
            normalized,
        )
        logger.debug("Session model path: %s" % session_model_path)
        dynamic_routing_task.update_session_from_rig(
            session_model_path,
            rig_model_path,
            session_model_path,
        )
    else:
        raise Exception("Unexpected modality: %s" % modality)

    return None


def upload(
    session: np_session.Session,
    recording_dirs: typing.Iterable[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    test: bool = False,
    hpc_upload_job_email: str = np_codeocean_upload.HPC_UPLOAD_JOB_EMAIL,
) -> None:
    """Writes and updates aind-data-schema to the session directory
     associated with the `session`. The aind-data-schema session model is
     updated to reflect the `rig_id` of the rig model added to the session
     directory.
    
    Determining which rig model to add depends on modality.

    If session modality is `ecephys`:
        - scrape the session model from the session directory
        - use the `rig_id` from the session model

    If session modality is `behavior`:
        - scrape the task output from the session directory
        - infer the `rig_id` from task output
    """
    modality = 'ecephys' \
        if np_codeocean_upload.is_ephys_session(session) else 'behavior'
    logger.debug(f"Modality: {modality}")
    rig_storage_directory = np_codeocean_upload.CONFIG["rig_metadata_dir"]
    logger.debug(f"Rig storage directory: {rig_storage_directory}")
    add_metadata(
        session.npexp_path,
        session.date,
        modality,
        rig_storage_directory,
    )
    return np_codeocean_upload.upload_session(
        session,
        recording_dirs=recording_dirs,
        force=force,
        dry_run=dry_run,
        test=test,
        hpc_upload_job_email=hpc_upload_job_email,
    )


def main() -> None:
    args = np_codeocean_upload.parse_args()
    main(**vars(args))


if __name__ == '__main__':
    main()