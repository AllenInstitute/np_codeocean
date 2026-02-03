import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Union

from aind_data_schema import base
from h5py import Dataset, File

logger = logging.getLogger(__name__)


def load_hdf5(h5_path: Path) -> File:
    """Load hdf5 file from path."""
    return File(h5_path, "r")


def extract_hdf5_value(h5_file: File, path: list[str]) -> Union[Any, None]:
    """Extract value from hdf5 file using a path. Path is a list of property
    names that are used to traverse the hdf5 file. A path of length greater
    than 1 is expected to point to a nested property.
    """
    try:
        value = None
        for part in path:
            value = h5_file[part]
    except KeyError as e:
        logger.warning(f"Key not found: {e}")
        return None

    if isinstance(value, Dataset):
        return value[()]
    else:
        return value


def find_replace_or_append(
    iterable: list[Any],
    filters: list[tuple[str, Any]],
    update: Any,
) -> None:
    """Find an item in a list of items that matches the filters and replace it.
    If no item is found, append.
    """
    for idx, obj in enumerate(iterable):
        if all(
            getattr(obj, prop_name, None) == prop_value
            for prop_name, prop_value in filters
        ):
            iterable[idx] = update
            break
    else:
        iterable.append(update)


def save_aind_model(
    model: base.AindCoreModel,
    output_path: Path,
) -> Path:
    """Save aind models to a specified output path.

    Notes
    -----
    - Gets around awkward `write_standard_file` method. to write to a determined
    output path.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        model.write_standard_file(temp_dir)
        return Path(
            shutil.copy2(Path(temp_dir) / model.default_filename(), output_path)
        )


# def save_aind_model_to_output_path(
#     model: base.AindCoreModel,
#     output_path: Path,
# ) -> Path:
#     """Convenience function that saves aind models to a specified output path.

#     Notes
#     -----
#     - Will soon replace `save_aind_model`.
#     - Mainly just gets around awkward `write_standard_file` method.
#     - Can raise an Exception if the output path is not supported. As of
#      aind-data-schema==0.33.3, the prefix and default filename must be
#      separated by an underscore.
#     """
#     output_directory = output_path.parent
#     logger.debug(f"output_directory: {output_directory}")
#     base_name = model.default_filename().replace(model._FILE_EXTENSION, "")
#     split = output_path.stem.split(base_name, maxsplit=1)
#     logger.debug("Parsed output filename parts: %s" % split)
#     if len(split) == 0:
#         prefix = None
#         suffix = None
#     elif len(split) == 1:
#         prefix = split[0]
#         suffix = None
#     elif len(split) == 2:
#         prefix = split[0]
#         suffix = split[1]
#     else:
#         raise Exception(
#             "Output path not supported. Output filename must be of the form: "
#             "<PREFIX>_<MODEL_NAME><SUFFIX>"
#         )

#     if prefix and not prefix.endswith("_"):
#         raise Exception(
#             "Invalid prefix, prefixes must end with an underscore.")

#     model.write_standard_file(
#         output_directory, prefix=prefix, suffix=suffix)
#     return output_path

# NEUROPIXELS_RIG_ROOM_MAP = {
#     "NP0": "325",
#     "NP1": "325",
#     "NP2": "327",
#     "NP3": "342",
# }

# BEHAVIOR_CLUSTER_ROOM_MAP = {
#     "B": "342",
#     "F": "346",
#     "G": "346",
#     "D": "347",
#     "E": "347",
# }


# def _get_rig_room(rig_name: str, room_map: dict[str, str]) -> Optional[str]:
#     try:
#         return room_map[rig_name]
#     except KeyError:
#         logger.debug("No room found for rig: %s" % rig_name)
#         return None


# def get_rig_room(rig_name: str) -> typing.Union[str, None]:
#     if rigs.is_behavior_box(rig_name):
#         return _get_rig_room(rig_name[0], BEHAVIOR_CLUSTER_ROOM_MAP)
#     else:
#         return _get_rig_room(rig_name, NEUROPIXELS_RIG_ROOM_MAP)


# def is_retrofitted_rig(rig_name: str) -> bool:
#     """Retrofitted rigs are behavior box rigs that have a speaker attached and
#      have a different solenoid.
#     """
#     return not rig_name.startswith("G")
