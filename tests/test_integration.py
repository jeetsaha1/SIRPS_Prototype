import os

from common.io import load
from common.schemas import FILES


def test_all_schema_files_exist_in_real_or_mock():
    os.environ["SIRPS_USE_MOCK"] = "1"
    for name in FILES:
        frame = load(name)
        assert len(frame) > 0
        assert frame.attrs["using_mock"] is True
