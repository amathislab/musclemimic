

from loco_mujoco.smpl import retargeting as retargeting_module


def test_amass_cache_miss_keeps_remote_download_fallback(monkeypatch, tmp_path):
    requested = []
    downloaded_path = tmp_path / "downloaded.npz"

    def record_download(**kwargs):
        requested.append(kwargs)
        return downloaded_path

    monkeypatch.setattr(retargeting_module, "try_download_gmr_cache", record_download)

    resolved = retargeting_module._prepare_gmr_cache_path(
        cache_path=str(tmp_path / "missing.npz"),
        env_name="MyoFullBody",
        dataset_name="KIT/sample",
        cache_root=str(tmp_path),
        clear_cache=False,
        allow_remote_download=True,
        logger=None,
    )

    assert resolved == str(downloaded_path)
    assert requested == [
        {
            "dataset_name": "KIT/sample",
            "env_name": "MyoFullBody",
            "cache_dir": str(tmp_path),
            "logger_override": None,
        }
    ]
