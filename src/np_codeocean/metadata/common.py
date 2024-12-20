
"""Models and controlled language for aind-data-schema devices names.
"""

import typing

import pydantic

# names of neuropixels rigs
RigName = typing.Literal["NP0", "NP1", "NP2", "NP3"]


class BehaviorBoxId(pydantic.BaseModel):
    """There is a divergence between the task rig id and the mpe rig id."""

    task_rig_id: RigName
    mpe_rig_id: str


DEFAULT_MVR_MAPPING = {
    "Camera 1": "Side",
    "Camera 2": "Eye",
    "Camera 3": "Front",
}


class ManipulatorInfo(pydantic.BaseModel):

    assembly_name: str
    serial_number: str


MODIFICATION_DATE_FORMAT = "%Y%m%d"
