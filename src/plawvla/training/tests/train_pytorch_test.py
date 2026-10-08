import importlib.util
import pathlib
import types

import pytest
import safetensors.torch
import torch

# This file directly tests the trainer's private checkpoint validation helper.
# ruff: noqa: SLF001


def _load_trainer():
    path = pathlib.Path(__file__).resolve().parents[4] / "scripts/train/train_pytorch.py"
    spec = importlib.util.spec_from_file_location("train_pytorch_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.foundation = torch.nn.Linear(2, 2, bias=False)
        self.world_model_adapter = torch.nn.Linear(2, 2, bias=False)


def _save(path: pathlib.Path, state: dict[str, torch.Tensor]) -> None:
    safetensors.torch.save_file({key: value.detach().clone() for key, value in state.items()}, path)


def test_foundation_weight_mode_allows_only_missing_world_model_tensors(tmp_path):
    trainer = _load_trainer()
    source = _TinyModel()
    target = _TinyModel()
    world_before = target.world_model_adapter.weight.detach().clone()
    checkpoint = tmp_path / "foundation.safetensors"
    _save(checkpoint, {"foundation.weight": source.foundation.weight})

    trainer._load_pytorch_weights(
        target,
        str(checkpoint),
        training_stage="post_training",
        weight_load_mode="foundation",
    )

    torch.testing.assert_close(target.foundation.weight, source.foundation.weight)
    torch.testing.assert_close(target.world_model_adapter.weight, world_before)

    with pytest.raises(ValueError, match="Missing checkpoint tensors"):
        trainer._load_pytorch_weights(
            _TinyModel(),
            str(checkpoint),
            training_stage="post_training",
            weight_load_mode="full",
        )


def test_foundation_weight_mode_rejects_stage_checkpoint(tmp_path):
    trainer = _load_trainer()
    checkpoint = tmp_path / "stage.safetensors"
    _save(checkpoint, _TinyModel().state_dict())

    with pytest.raises(ValueError, match="without PLaW-VLA world-model tensors"):
        trainer._load_pytorch_weights(
            _TinyModel(),
            str(checkpoint),
            training_stage="wm_alignment",
            weight_load_mode="foundation",
        )

    trainer._load_pytorch_weights(
        _TinyModel(),
        str(checkpoint),
        training_stage="post_training",
        weight_load_mode="full",
    )


def test_initialization_metadata_records_resolved_source_without_local_path():
    trainer = _load_trainer()
    config = types.SimpleNamespace(
        initialization_source_requested="auto",
        initialization_source_resolved="pi05",
        weight_load_mode="foundation",
        pytorch_weight_path="/machine/local/checkpoint",
    )

    metadata = trainer._initialization_metadata(config)

    assert metadata == {
        "requested_source": "auto",
        "resolved_source": "pi05",
        "weight_load_mode": "foundation",
    }
    assert "/machine/local/checkpoint" not in repr(metadata)


def test_resume_restores_original_initialization_metadata(tmp_path):
    trainer = _load_trainer()
    config = types.SimpleNamespace(
        initialization_source_requested=None,
        initialization_source_resolved=None,
        weight_load_mode="full",
    )
    torch.save(
        {
            "initialization": {
                "requested_source": "auto",
                "resolved_source": "pi05",
                "weight_load_mode": "foundation",
            }
        },
        tmp_path / "metadata.pt",
    )

    trainer._restore_initialization_metadata(config, tmp_path)

    assert config.initialization_source_requested == "auto"
    assert config.initialization_source_resolved == "pi05"
    assert config.weight_load_mode == "foundation"
