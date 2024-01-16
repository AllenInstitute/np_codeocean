import os
import pathlib

import dotenv
import np_session
import pytest

from np_codeocean import aind_data

dotenv.load_dotenv()

session_id_var_name = "NP_SESSION_ID"
try:
    session_id = os.environ[session_id_var_name]
except KeyError as exc:
    raise Exception("%s required testing." % session_id_var_name) from exc


@pytest.mark.onprem
def test_aind_dataschema(tmpdir):
    aind_data.generate_rig(
        np_session.Session(session_id),
        pathlib.Path("./tests/resources/rig.json"),
        tmpdir,
    )

    assert (tmpdir / "rig.json").isfile()