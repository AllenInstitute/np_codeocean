import pytest
import json
import pathlib
import datetime
from np_codeocean.metadata import storage


@pytest.fixture
def storage_fixture(
    tmp_path,
):
    storage_directory = tmp_path / "storage"
    storage_directory.mkdir()
    yield storage_directory
    storage_directory.rmdir()


def test_storage_update_get_update_get(
    tmp_path: pathlib.Path
) -> None:
    item_0 = {
        "rig_id": "327_NP1_240401",
    }
    item_0_path = tmp_path / "item-0.json"
    item_0_path.write_text(json.dumps(item_0))
    item_0_timestamp = datetime.datetime(
        2024, 4, 1
    )
    tags = ("NP1", )
    initial = storage.get_item(
        tmp_path,
        item_0_timestamp,
        *tags,
    )
    assert initial is None, "No initial item."
    storage.update_item(
        tmp_path,
        item_0_path,
        item_0_timestamp,
        *tags,
    )
    stored_0_path = storage.get_item(
        tmp_path,
        item_0_timestamp,
        *tags,
    )
    assert stored_0_path
    stored_item = json.loads(
        stored_0_path.read_text())
    assert stored_item == item_0, \
        "Stored item is what we expect."
    item_1 = {
        "rig_id": "327_NP1_240402",
    }
    item_1_path = tmp_path / "item-1.json"
    item_1_path.write_text(json.dumps(item_1))
    item_1_timestamp = datetime.datetime(
        2024, 4, 2
    )
    storage.update_item(
        tmp_path,
        item_1_path,
        item_1_timestamp,
        *tags,
    )
    stored_1_path = storage.get_item(
        tmp_path,
        item_0_timestamp,
        *tags,
    )
    assert stored_1_path
    updated_item = json.loads(stored_1_path.read_text())
    assert updated_item == item_0, \
        "Stored item is not previous."
    assert updated_item != item_1, "Stored item is updated item."

    item_2 = {
        "rig_id": "327_NP1_240302",
    }
    item_2_path = tmp_path / "item-2.json"
    item_2_path.write_text(json.dumps(item_2))
    item_2_timestamp = datetime.datetime(
        2024, 3, 2
    )
    storage.update_item(
        tmp_path,
        item_2_path,
        item_2_timestamp,
        *tags,
    )
    stored_2_path = storage.get_item(
        tmp_path,
        item_2_timestamp,
        *tags,
    )
    assert stored_2_path
    updated_item = json.loads(stored_2_path.read_text())
    assert updated_item != item_0
    assert updated_item == item_2
