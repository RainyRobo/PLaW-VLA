import json

import numpy as np

from openpi.shared import normalize
from openpi.training import checkpoints
from openpi.training import config as _config


def _make_norm_stats(offset: float = 0.0) -> dict[str, normalize.NormStats]:
    stats = normalize.RunningStats()
    stats.update(np.arange(12).reshape(4, 3) + offset)
    return {"actions": stats.get_statistics()}


def test_save_norm_stats_assets_normalizes_checkpoint_paths(tmp_path):
    norm_stats = _make_norm_stats()

    checkpoints.save_norm_stats_assets(
        tmp_path,
        norm_stats,
        ["/datasets/libero_v3_eef", "physical-intelligence/libero"],
    )

    assert (tmp_path / "datasets/libero_v3_eef/norm_stats.json").exists()
    assert (tmp_path / "physical-intelligence/libero/norm_stats.json").exists()


def test_load_norm_stats_normalizes_absolute_asset_ids(tmp_path):
    norm_stats = _make_norm_stats()
    checkpoints.save_norm_stats_assets(tmp_path, norm_stats, "/datasets/libero_v3_eef")

    loaded = checkpoints.load_norm_stats(tmp_path, ["/datasets/libero_v3_eef"])

    assert np.allclose(loaded["actions"].mean, norm_stats["actions"].mean)
    assert np.allclose(loaded["actions"].std, norm_stats["actions"].std)


def test_save_data_configs_assets_writes_all_assets_and_manifest(tmp_path):
    norm_stats = _make_norm_stats()
    data_configs = [
        _config.DataConfig(
            repo_id=["/datasets/libero_a"],
            asset_id=["/datasets/libero_a"],
            norm_stats=norm_stats,
        ),
        _config.DataConfig(
            repo_id="libero-100",
            asset_id="libero-100",
            norm_stats=norm_stats,
        ),
    ]

    checkpoints.save_data_configs_assets(
        tmp_path,
        data_configs,
        checkpoint_asset_metadata=[
            {
                "dataset_type": "libero",
                "repo_id": data_configs[0].repo_id,
                "asset_id": data_configs[0].asset_id,
                "weight": 0.1,
            },
            {
                "dataset_type": "libero",
                "repo_id": data_configs[1].repo_id,
                "asset_id": data_configs[1].asset_id,
                "weight": 1.0,
            },
        ],
    )

    assert (tmp_path / "datasets/libero_a/norm_stats.json").exists()
    assert (tmp_path / "libero-100/norm_stats.json").exists()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["datasets"][0]["dataset_type"] == "libero"
    assert manifest["datasets"][0]["weight"] == 0.1
    assert manifest["datasets"][0]["checkpoint_asset_ids"] == ["datasets/libero_a"]
    assert manifest["datasets"][0]["has_per_repo_norm_stats"] is False
    assert manifest["datasets"][0]["per_repo_norm_stats_count"] == 0
    assert manifest["datasets"][1]["dataset_type"] == "libero"
    assert manifest["datasets"][1]["weight"] == 1.0
    assert manifest["datasets"][1]["checkpoint_asset_ids"] == ["libero-100"]


def test_save_data_configs_assets_fans_out_per_repo_norm_stats(tmp_path):
    """Multi-task multi-repo configs must write each task's own stats.

    Regression test for the bug where a multi-dataset checkpoint replicated the
    single ``norm_stats`` field across every ``asset_id``, causing every task
    after the first to be normalised with the wrong stats at inference.
    """
    repo_ids = [
        "/data/libero_v3/task_001",
        "/data/libero_v3/task_002",
        "/data/libero_v3/task_003",
    ]
    asset_ids = list(repo_ids)
    per_repo_norm_stats = {
        repo_ids[0]: _make_norm_stats(offset=0.0),
        repo_ids[1]: _make_norm_stats(offset=100.0),
        repo_ids[2]: _make_norm_stats(offset=-50.0),
    }

    data_configs = [
        _config.DataConfig(
            repo_id=repo_ids,
            asset_id=asset_ids,
            # `norm_stats` mirrors the loader behaviour (first task's stats).
            norm_stats=per_repo_norm_stats[repo_ids[0]],
            per_repo_norm_stats=per_repo_norm_stats,
        ),
    ]

    checkpoints.save_data_configs_assets(tmp_path, data_configs)

    # Each task must end up with its OWN stats on disk, not the parent fallback.
    for repo_id in repo_ids:
        expected = per_repo_norm_stats[repo_id]["actions"]
        # The repo_ids in this test are absolute paths; the checkpoint asset
        # layout strips the leading slash but keeps the rest verbatim.
        on_disk_dir = tmp_path / repo_id.lstrip("/")
        loaded = normalize.load(on_disk_dir)
        assert np.allclose(loaded["actions"].mean, expected.mean), repo_id
        assert np.allclose(loaded["actions"].std, expected.std), repo_id

    # Sanity: the three on-disk files must not be byte-identical (which is the
    # exact symptom the pre-fix code produced).
    on_disk_means = {
        repo_id: normalize.load(tmp_path / repo_id.lstrip("/"))["actions"].mean.tolist()
        for repo_id in repo_ids
    }
    assert len({tuple(v) for v in on_disk_means.values()}) == len(repo_ids)


def test_save_data_configs_assets_falls_back_when_per_repo_missing(tmp_path):
    """Without per-repo stats, every asset_id gets the same (single) norm_stats."""
    norm_stats = _make_norm_stats()
    data_config = _config.DataConfig(
        repo_id=["/repo/a", "/repo/b"],
        asset_id=["/repo/a", "/repo/b"],
        norm_stats=norm_stats,
        per_repo_norm_stats=None,
    )

    checkpoints.save_data_configs_assets(tmp_path, [data_config])

    for asset_id in ("repo/a", "repo/b"):
        loaded = normalize.load(tmp_path / asset_id)
        assert np.allclose(loaded["actions"].mean, norm_stats["actions"].mean)


def test_save_data_configs_assets_uses_norm_stats_fallback_for_unknown_repo(tmp_path):
    """A repo_id missing from per_repo_norm_stats must fall back to data_config.norm_stats."""
    fallback = _make_norm_stats(offset=0.0)
    a_stats = _make_norm_stats(offset=10.0)
    data_config = _config.DataConfig(
        repo_id=["/repo/a", "/repo/b"],
        asset_id=["/repo/a", "/repo/b"],
        norm_stats=fallback,
        per_repo_norm_stats={"/repo/a": a_stats},
    )

    checkpoints.save_data_configs_assets(tmp_path, [data_config])

    a_loaded = normalize.load(tmp_path / "repo/a")
    b_loaded = normalize.load(tmp_path / "repo/b")
    assert np.allclose(a_loaded["actions"].mean, a_stats["actions"].mean)
    assert np.allclose(b_loaded["actions"].mean, fallback["actions"].mean)
