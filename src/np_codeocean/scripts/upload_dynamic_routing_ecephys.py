import argparse
import contextlib
import datetime
import logging
import pathlib
import time
import typing
import warnings

import np_config
import npc_session
import npc_sessions
from aind_data_schema.core.rig import Rig
from aind_data_schema.core.session import Session as AindSession
import aind_codeocean_pipeline_monitor.models 
import codeocean.capsule
import codeocean.data_asset
import codeocean.computation 
import np_codeocean
from np_codeocean.metadata import core as metadata_core
from aind_data_schema_models.modalities import Modality

# Disable divide by zero or NaN warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    filename=f"//allen/programs/mindscope/workgroups/np-exp/codeocean-logs/{pathlib.Path(__file__).stem}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
    level=logging.DEBUG, 
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s", 
    datefmt="%Y-%d-%m %H:%M:%S",
    )
logger = logging.getLogger(__name__)

CONFIG = np_config.fetch('/rigs/room_numbers')


def reformat_rig_model_rig_id(rig_id: str, modification_date: datetime.date) -> str:
    rig_record = npc_session.RigRecord(rig_id)
    if not rig_record.is_neuro_pixels_rig:
        raise Exception(
            f"Rig is not a neuropixels rig. Only behavior cluster rigs are supported. rig_id={rig_id}")
    room_number = CONFIG.get(rig_record, "UNKNOWN")
    return rig_record.as_aind_data_schema_rig_id(str(room_number), modification_date).replace('.','')


def extract_modification_date(rig: Rig) -> datetime.date:
    _, _, date_str = rig.rig_id.split("_")
    if len(date_str) == 6:
        return datetime.datetime.strptime(date_str, "%y%m%d").date()
    elif len(date_str) == 8:
        return datetime.datetime.strptime(date_str, "%Y%m%d").date()
    else:
        raise Exception(f"Unsupported date format: {date_str}")


def add_metadata(
    session_directory: str | pathlib.Path,
    session_datetime: datetime.datetime,
    rig_storage_directory: pathlib.Path,
    ignore_errors: bool = True,
    skip_existing: bool = True,
) -> None:
    """Adds rig and sessions metadata to a session directory.
    """
    normalized_session_dir = np_config.normalize_path(session_directory)
    logger.debug(f"{normalized_session_dir = }")
    logger.debug(f"{rig_storage_directory = }")
    session_json = normalized_session_dir / "session.json"
    if not skip_existing or not (session_json.is_symlink() or session_json.exists()):
        logger.debug("Attempting to create session.json")
        npc_sessions.DynamicRoutingSession(normalized_session_dir)._aind_session_metadata.write_standard_file(normalized_session_dir)
        if session_json.exists():
            logger.debug("Created session.json")
        else:
            raise FileNotFoundError("Failed to find created session.json, but no error occurred during creation: may be in unexpected location")
    _ = AindSession.model_validate_json(session_json.read_text())
    
    rig_model_path = normalized_session_dir / "rig.json"
    if not skip_existing or not (rig_model_path.is_symlink() or rig_model_path.exists()):
        if not (session_json.is_symlink() or session_json.exists()):
            logger.warning("session.json is currently required for the rig.json to be created, so we can't continue with metadata creation")
            return None
        metadata_core.add_np_rig_to_session_dir(
            normalized_session_dir,
            session_datetime,
            rig_storage_directory,
        )
        if rig_model_path.exists():
            logger.debug("Created rig.json")
        else:
            raise FileNotFoundError("Failed to find created rig.json, but no error occurred during creation: may be in unexpected location")
    if not (rig_model_path.is_symlink() or rig_model_path.exists()):
        return None

    rig_metadata = Rig.model_validate_json(rig_model_path.read_text())
    modification_date = extract_modification_date(rig_metadata)
    rig_metadata.rig_id = reformat_rig_model_rig_id(rig_metadata.rig_id, modification_date)
    rig_metadata.write_standard_file(normalized_session_dir)  # assumes this will work out to dest/rig.json
    session_model_path = metadata_core.scrape_session_model_path(
        normalized_session_dir,
    )
    metadata_core.update_session_from_rig(
        session_model_path,
        rig_model_path,
        session_model_path,
    )

    return None


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
        session_datetime=(
            session.start 
            if not np_codeocean.is_surface_channel_recording(session.npexp_path.name)
            else np_codeocean.get_surface_channel_start_time(session)
        ),
        rig_storage_directory=pathlib.Path(np_codeocean.get_project_config()["rig_metadata_dir"]),
        ignore_errors=True,
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
                        id="", # ID of new raw data asset will be inserted here by airflow
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
                tags=[str(session.mouse), 'derived', 'ecephys'],
                custom_metadata={'data level': 'derived', 'experiment type': 'ecephys', 'subject id': str(session.mouse)},
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
    parser.add_argument('session_path_or_folder_name', help="session ID (lims or np-exp foldername) or path to session folder")
    parser.add_argument('recording_dirs', nargs='*', help="[optional] specific names of recording directories to upload - for use with split recordings only.")
    parser.add_argument('--email', dest='hpc_upload_job_email', type=str, help=f"[optional] specify email address for hpc upload job updates. Default is {np_codeocean.HPC_UPLOAD_JOB_EMAIL}")
    parser.add_argument('--force', action='store_true', help="enable `force_cloud_sync` option, re-uploading and re-making raw asset even if data exists on S3")
    parser.add_argument('--test', action='store_true', help="use the test-upload service, uploading to the test CodeOcean server instead of the production server")
    parser.add_argument('--dry-run', action='store_true', help="Create upload job but do not submit to hpc upload queue.")
    parser.add_argument('--preserve-symlinks', dest='regenerate_symlinks', action='store_false', help="Existing symlink folders will not be deleted and regenerated - may result in additional data being uploaded")
    parser.add_argument('--regenerate-metadata', action='store_true', help="Regenerate metadata files (session.json and rig.json) even if they already exist")
    parser.add_argument('--sync', dest="adjust_ephys_timestamps", action='store_true', help="Adjust ephys timestamps.npy prior to upload using sync data (if available)")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    kwargs = vars(args)
    np_codeocean.utils.set_npc_lims_credentials()
    write_metadata_and_upload(**kwargs)


if __name__ == '__main__':
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