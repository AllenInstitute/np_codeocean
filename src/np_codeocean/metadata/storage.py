"""Stores a history of object changes in local storage. Simple updater and
 getter for iterations of a file.
"""

import datetime
import json
import logging
import pathlib
import shutil

logger = logging.getLogger(__name__)


def _generate_item_filename(
    *tags: str,
) -> str:
    return "_".join(str(tag) for tag in tags) + "_rig.json"


TIMESTAMP_FORMAT = "%Y-%m-%d-%H-%M-%S"


def _item_path_sorter(path: pathlib.Path) -> datetime.datetime:
    mtime_str = path.stem.replace("_rig", "").split("_")[-1]
    return datetime.datetime.strptime(mtime_str, TIMESTAMP_FORMAT)


def get_item(
    storage_directory: pathlib.Path,
    timestamp: datetime.datetime,
    *_tags: str,
) -> pathlib.Path | None:
    """Gets latest item before or equal to timestamp."""
    search_pattern = _generate_item_filename(*_tags, "*")
    logger.info("Search pattern: %s" % search_pattern)
    items = list(storage_directory.glob(search_pattern))
    if not items:
        logger.debug("No item found for tags: %s" % json.dumps(_tags))
        return None

    sorted_items = sorted(items, key=_item_path_sorter)
    logger.debug("Fetched items: %s" % sorted_items)
    logger.debug([_item_path_sorter(item) for item in sorted_items])
    filtered = list(
        filter(
            lambda item: _item_path_sorter(item) <= timestamp,
            sorted_items,
        )
    )
    logger.debug("Filtered items: %s" % filtered)
    if not filtered:
        return None
    return filtered[-1]


def update_item(
    storage_directory: pathlib.Path,
    filepath: pathlib.Path,
    timestamp: datetime.datetime,
    *_tags: str,
) -> pathlib.Path:
    """
    Notes
    -----
    - Also used to initialize a new base rig in local storage.
    - If a rig with the same name doest not exist, a new rig will be created.
    """
    filename = _generate_item_filename(
        *_tags,
        timestamp.strftime(TIMESTAMP_FORMAT),
    )

    return pathlib.Path(
        shutil.copy2(
            filepath,
            storage_directory / filename,
        )
    )
