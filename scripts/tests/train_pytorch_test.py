import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from .. import train_pytorch


def test_build_log_payload_averages_scalar_metrics():
    infos = [
        {
            "loss": 10.0,
            "learning_rate": 1e-4,
            "grad_norm": 2.0,
            "wm_loss": 1.0,
        },
        {
            "loss": 14.0,
            "learning_rate": 3e-4,
            "grad_norm": 4.0,
            "wm_loss": 3.0,
        },
    ]

    payload = train_pytorch._build_log_payload(
        infos,
        global_step=20,
        elapsed=8.0,
        log_interval=4,
    )

    assert payload["loss"] == 12.0
    assert payload["learning_rate"] == pytest.approx(2e-4)
    assert payload["grad_norm"] == 3.0
    assert payload["time_per_step"] == 2.0
    assert payload["wm_loss"] == 2.0
