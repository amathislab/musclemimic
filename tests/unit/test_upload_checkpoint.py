import json

import scripts.upload_checkpoint as upload_module


def _make_checkpoint(tmp_path, metadata=None):
    checkpoint = tmp_path / "checkpoint_123"
    checkpoint.mkdir()
    (checkpoint / "train_state").mkdir()
    (checkpoint / "config").mkdir()
    (checkpoint / "metadata").mkdir()
    (checkpoint / "_CHECKPOINT_METADATA").touch()
    if metadata is not None:
        (checkpoint / "metadata" / "metadata").write_text(json.dumps(metadata))
    return checkpoint


def test_download_stats_config_contains_policy_metadata(tmp_path):
    checkpoint = _make_checkpoint(
        tmp_path,
        {
            "algo_version": "PPOJax_v1",
            "schema_version": "2.2",
            "backend": "jax",
            "env_name": "MyoFullBody",
            "global_timestep": 123456,
        },
    )

    config = json.loads(upload_module._build_download_stats_config(checkpoint))

    assert config == {
        "model_type": "musclemimic-policy",
        "checkpoint_format": "orbax",
        "checkpoint_name": "checkpoint_123",
        "algo_version": "PPOJax_v1",
        "schema_version": "2.2",
        "backend": "jax",
        "env_name": "MyoFullBody",
        "global_timestep": 123456,
    }


def test_upload_checkpoint_adds_root_config_json_for_hf_stats(tmp_path, monkeypatch):
    checkpoint = _make_checkpoint(tmp_path, {"env_name": "MyoBimanualArm"})
    calls = []

    class FakeApi:
        def upload_folder(self, **kwargs):
            calls.append(("folder", kwargs))

        def upload_file(self, **kwargs):
            calls.append(("file", kwargs))

    monkeypatch.setattr(upload_module, "HfApi", FakeApi)
    monkeypatch.setattr(upload_module, "create_repo", lambda *args, **kwargs: None)

    upload_module.upload_checkpoint(checkpoint, "org/policy")

    assert calls[0] == (
        "folder",
        {
            "folder_path": str(checkpoint),
            "repo_id": "org/policy",
            "repo_type": "model",
        },
    )
    assert calls[1][0] == "file"
    assert calls[1][1]["path_in_repo"] == "config.json"
    assert calls[1][1]["repo_id"] == "org/policy"
    assert calls[1][1]["repo_type"] == "model"
    uploaded_config = json.loads(calls[1][1]["path_or_fileobj"])
    assert uploaded_config["model_type"] == "musclemimic-policy"
    assert uploaded_config["env_name"] == "MyoBimanualArm"
