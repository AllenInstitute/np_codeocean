import argparse
import datetime
import logging
import pathlib
import time
import typing
import warnings
from typing import Callable

import aind_codeocean_pipeline_monitor.models
import aind_data_schema.base
import codeocean.computation
import codeocean.data_asset
import np_config
import npc_session
import npc_sessions
import npc_sessions.aind_data_schema
from aind_data_schema_models.modalities import Modality

import np_codeocean

# Disable divide by zero or NaN warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    filename=f"//allen/programs/mindscope/workgroups/np-exp/codeocean-logs/{pathlib.Path(__file__).stem}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
    level=logging.DEBUG,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG = np_config.fetch("/rigs/room_numbers")


def reformat_rig_model_rig_id(rig_id: str, modification_date: datetime.date) -> str:
    rig_record = npc_session.RigRecord(rig_id)
    if not rig_record.is_neuro_pixels_rig:
        raise Exception(
            f"Rig is not a neuropixels rig. Only behavior cluster rigs are supported. rig_id={rig_id}"
        )
    room_number = CONFIG.get(rig_record, "UNKNOWN")
    return rig_record.as_aind_data_schema_rig_id(
        str(room_number), modification_date
    ).replace(".", "")


def add_metadata(
    session_directory: str | pathlib.Path,
    ignore_errors: bool = False,
    skip_existing: bool = True,
) -> None:
    """Adds rig and sessions metadata to a session directory."""
    normalized_session_dir = np_config.normalize_path(session_directory)
    fname_to_fn: dict[
        str, Callable[[npc_sessions.Session], aind_data_schema.base.DataCoreModel]
    ] = {
        "acquisition.json": npc_sessions.aind_data_schema.get_acquisition_model,
        "instrument.json": npc_sessions.aind_data_schema.get_instrument_model,
        "data_description.json": npc_sessions.aind_data_schema.get_data_description_model,
    }
    if skip_existing and all(
        (normalized_session_dir / fname).exists() for fname in fname_to_fn
    ):
        # exit before making a session object if we don't need to
        logger.info(f"{len(fname_to_fn)} required metadata files exist. Skipping.")
        return None
    logger.debug(f"Trying to create npc_sessions.Session({normalized_session_dir})")
    try:
        if "_surface_channels" in normalized_session_dir.as_posix():
            session = npc_sessions.DynamicRoutingSurfaceRecording(
                normalized_session_dir, is_sync=False
            )
        else:
            session = npc_sessions.DynamicRoutingSession(normalized_session_dir)
    except Exception as e:
        logger.error(f"Error creating npc_sessions.DynamicRoutingSession: {e!r}")
        if ignore_errors:
            return None
        raise
    for fname, fn in fname_to_fn.items():
        path = normalized_session_dir / fname
        if not skip_existing or not path.exists():
            try:
                model = fn(session)
            except Exception as e:
                logger.error(f"Error creating model for {fname}: {e!r}")
                if ignore_errors:
                    continue
                raise
            else:
                model.write_standard_file(normalized_session_dir)
                logger.info(f"Wrote {fname}")


def write_metadata_and_upload(
    session_path_or_folder_name: str,
    recording_dirs: typing.Iterable[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    test: bool = False,
    hpc_upload_job_email: str = np_codeocean.HPC_UPLOAD_JOB_EMAIL,
    regenerate_metadata: bool = False,
    regenerate_symlinks: bool = True,
    adjust_ephys_timestamps: bool = False,
) -> None:
    """Writes and updates aind-data-schema to the session directory
     associated with the `session`. The aind-data-schema session model is
     updated to reflect the `rig_id` of the rig model added to the session
     directory.

    Only handles ecephys platform uploads (ie sessions with a folder of data; not
    behavior box sessions, which have a single hdf5 file only)
    """
    # session = np_session.Session(session) #! this doesn't work for surface_channels
    session = np_codeocean.get_np_session(session_path_or_folder_name)
    add_metadata(
        session_directory=session.npexp_path,
        ignore_errors=False,
        skip_existing=not regenerate_metadata,
    )

    # Optional codeocean_pipeline_settings as {modality_abbr: PipelineMonitorSettings}
    # You can specify up to one pipeline conf per modality
    # In the future, these can be stored in AWS param store as part of a "job_type"
    codeocean_pipeline_settings = {
        Modality.ECEPHYS.abbreviation: aind_codeocean_pipeline_monitor.models.PipelineMonitorSettings(
            run_params=codeocean.computation.RunParams(
                capsule_id="287db808-74ce-4e44-b14b-fde1471eba45",
                data_assets=[
                    codeocean.data_asset.DataAsset(
                        name="",
                        id="",  # ID of new raw data asset will be inserted here by airflow
                        mount="ecephys",
                        created=time.time(),
                        state=codeocean.data_asset.DataAssetState.Draft,
                        type=codeocean.data_asset.DataAssetType.Dataset,
                        last_used=time.time(),
                    ),
                ],
            ),
            computation_polling_interval=15 * 60,
            computation_timeout=48 * 3600,
            capture_settings=aind_codeocean_pipeline_monitor.models.CaptureSettings(
                tags=[str(session.mouse), "derived", "ecephys"],
                custom_metadata={
                    "data level": "derived",
                    "experiment type": "ecephys",
                    "subject id": str(session.mouse),
                },
                process_name_suffix="sorted",
                process_name_suffix_tz="US/Pacific",
            ),
        ),
    }

    return np_codeocean.upload_session(
        session_path_or_folder_name,
        recording_dirs=recording_dirs,
        force=force,
        dry_run=dry_run,
        test=test,
        hpc_upload_job_email=hpc_upload_job_email,
        regenerate_symlinks=regenerate_symlinks,
        adjust_ephys_timestamps=adjust_ephys_timestamps,
        codeocean_pipeline_settings=codeocean_pipeline_settings,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a session to CodeOcean")
    parser.add_argument(
        "session_path_or_folder_name",
        help="session ID (lims or np-exp foldername) or path to session folder",
    )
    parser.add_argument(
        "recording_dirs",
        nargs="*",
        help="[optional] specific names of recording directories to upload - for use with split recordings only.",
    )
    parser.add_argument(
        "--email",
        dest="hpc_upload_job_email",
        type=str,
        help=f"[optional] specify email address for hpc upload job updates. Default is {np_codeocean.HPC_UPLOAD_JOB_EMAIL}",
        default=np_codeocean.HPC_UPLOAD_JOB_EMAIL,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="enable `force_cloud_sync` option, re-uploading and re-making raw asset even if data exists on S3",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="use the test-upload service, uploading to the test CodeOcean server instead of the production server",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create upload job but do not submit to hpc upload queue.",
    )
    parser.add_argument(
        "--preserve-symlinks",
        dest="regenerate_symlinks",
        action="store_false",
        help="Existing symlink folders will not be deleted and regenerated - may result in additional data being uploaded",
    )
    parser.add_argument(
        "--regenerate-metadata",
        action="store_true",
        help="Regenerate metadata files (session.json and rig.json) even if they already exist",
    )
    parser.add_argument(
        "--sync",
        dest="adjust_ephys_timestamps",
        action="store_true",
        help="Adjust ephys timestamps.npy prior to upload using sync data (if available)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = vars(args)
    np_codeocean.utils.set_npc_lims_credentials()
    write_metadata_and_upload(**kwargs)


if __name__ == "__main__":
    main()
    # write_metadata_and_upload(
    #     'DRpilot_744740_20241113_surface_channels',
    #     force=False,
    #     regenerate_metadata=False,
    #     regenerate_symlinks=False,
    # )
    # upload_dr_ecephys DRpilot_712141_20240606 --regenerate-metadata
    # upload_dr_ecephys DRpilot_712141_20240611 recording1 recording2 --regenerate-metadata --force
    # upload_dr_ecephys DRpilot_712141_20240605 --regenerate-metadata
