from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCES = REPO_ROOT / "hackathon resources"

@pytest.fixture(scope="session")
def descriptions_csv():
    return RESOURCES / "Description_PROC.csv"

@pytest.fixture(scope="session")
def reviews_csv():
    return RESOURCES / "Reviews_PROC.csv"

@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "state.sqlite"
