import os

import pynvml
import pytest


def set_jax_cpu_backend_if_no_gpu() -> None:
    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        # No GPU found.
        os.environ["JAX_PLATFORMS"] = "cpu"


def pytest_configure(config: pytest.Config) -> None:
    set_jax_cpu_backend_if_no_gpu()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--dataset-config-name",
        action="store",
        default=None,
        help="Config name for manual dataset-load tests.",
    )


@pytest.fixture
def dataset_config_name(request: pytest.FixtureRequest) -> str | None:
    return request.config.getoption("dataset_config_name")
