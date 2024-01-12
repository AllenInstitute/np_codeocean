import os
import pytest
import pathlib
import dotenv
import np_session

from np_codeocean import aind_data

dotenv.load_dotenv()

session_id_var_name = "NP_SESSION_ID"
try:
    session_id = os.environ[session_id_var_name]
except KeyError:
    raise Exception("%s required testing.")


@pytest.mark.onprem
def test_aind_dataschema(tmpdir):
    aind_data.generate_rig(
        np_session.Session(session_id),
        pathlib.Path("./tests/resources/rig.json"),
        tmpdir,
    )

    assert (tmpdir / "rig.json").is_file()