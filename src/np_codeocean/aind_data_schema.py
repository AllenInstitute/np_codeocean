import logging
import np_session
import shutil
import tempfile
import pathlib
from aind_metadata_mapper.neuropixels import mvr_rig, sync_rig, \
    open_ephys_rig


logger = logging.getLogger(__name__)


def generate_rig_context_dir(
    session: np_session.Session,
    rig_json_path: pathlib.Path,
) -> pathlib.Path:
    rig_context_dir = pathlib.Path(tempfile.mkdtemp())
    # copy all open ephys settings files, rename to avoid naming clashes
    for open_ephys_settings_file in session.npexp_path.glob("**\settings.xml"):
        shutil.copy(
            open_ephys_settings_file,
            rig_context_dir / f'{open_ephys_settings_file.parts[-2]}.open_ephys.{open_ephys_settings_file.name}',
        )

    shutil.copy(rig_json_path, rig_context_dir)

    return rig_context_dir


def generate_rig(
    session: np_session.Session,
    rig_json_path: pathlib.Path,
    output_dir: pathlib.Path,
):
    rig_context_dir = generate_rig_context_dir(
        session,
        rig_json_path,
    )
    
    sync_resource_name = "sync.yml"
    if (rig_context_dir / sync_resource_name).is_file():
        sync_rig.SyncRigEtl(
            rig_context_dir,
            rig_context_dir,
            config_resource_name=sync_resource_name,
            sync_daq_name="Sync",
        )

    mvr_resource_name = "mvr.ini"
    if (rig_context_dir / mvr_resource_name).is_file():
        mvr_rig.MvrRigEtl(
            rig_context_dir,
            rig_context_dir,
            "127.0.0.1",
            {
                "Camera 1": "Behavior",
                "Camera 2": "Eye",
                "Camera 3": "Face forward",
            },
            mvr_resource_name=mvr_resource_name,
        )
    
    for open_ephys_settings in rig_context_dir.glob("*.open_ephys.settings.xml"):
        open_ephys_rig.OpenEphysRigEtl(
            rig_context_dir,
            rig_context_dir,
            open_ephys_settings_resource_name=open_ephys_settings.name,
        )

    shutil.copy(
        rig_context_dir / "rig.json",
        output_dir,
    )
