import pytest

from tests.sample_data import train_sample_model


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    train_sample_model(out)
    return out
