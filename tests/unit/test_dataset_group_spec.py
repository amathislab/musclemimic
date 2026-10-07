"""Tests for expand_amass_dataset_group_spec and get_amass_dataset_groups."""

import pytest
from omegaconf import ListConfig

from loco_mujoco.task_factories.dataset_confs import (
    AMASSDatasetConf,
    expand_amass_dataset_group_spec,
    get_amass_dataset_groups,
)


def test_amass_dataset_conf_normalizes_max_motions_seed():
    conf = AMASSDatasetConf(rel_dataset_path="motion", max_motions_seed="7")

    assert conf.max_motions_seed == 7
    assert isinstance(conf.max_motions_seed, int)


# --- expand_amass_dataset_group_spec ---


class TestExpandAmassDatasetGroupSpec:
    def test_none_returns_empty(self):
        assert expand_amass_dataset_group_spec(None) == []

    def test_single_string(self):
        assert expand_amass_dataset_group_spec("FOO") == ["FOO"]

    def test_plus_joined_string(self):
        result = expand_amass_dataset_group_spec("A + B + C")
        assert result == ["A", "B", "C"]

    def test_plus_joined_strips_whitespace(self):
        result = expand_amass_dataset_group_spec("  A +  B  ")
        assert result == ["A", "B"]

    def test_plus_joined_ignores_empty_segments(self):
        result = expand_amass_dataset_group_spec("A ++ B")
        assert result == ["A", "B"]

    def test_list_of_strings(self):
        result = expand_amass_dataset_group_spec(["X", "Y"])
        assert result == ["X", "Y"]

    def test_tuple_of_strings(self):
        result = expand_amass_dataset_group_spec(("X", "Y"))
        assert result == ["X", "Y"]

    def test_listconfig(self):
        result = expand_amass_dataset_group_spec(ListConfig(["A", "B"]))
        assert result == ["A", "B"]

    def test_nested_list_with_plus_joined(self):
        result = expand_amass_dataset_group_spec(["A + B", "C"])
        assert result == ["A", "B", "C"]

    def test_nested_listconfig_with_plus_joined(self):
        result = expand_amass_dataset_group_spec(ListConfig(["X + Y", "Z"]))
        assert result == ["X", "Y", "Z"]

    def test_empty_string_returns_empty(self):
        assert expand_amass_dataset_group_spec("") == []

    def test_whitespace_only_string_returns_empty(self):
        assert expand_amass_dataset_group_spec("   ") == []

    def test_empty_list_returns_empty(self):
        assert expand_amass_dataset_group_spec([]) == []

    def test_invalid_type_raises_typeerror(self):
        with pytest.raises(TypeError, match="got int"):
            expand_amass_dataset_group_spec(42)

    def test_invalid_type_in_list_raises_typeerror(self):
        with pytest.raises(TypeError, match="got int"):
            expand_amass_dataset_group_spec(["A", 42])


# --- get_amass_dataset_groups ---


class TestGetAmassDatasetGroups:
    def test_returns_dict(self):
        groups = get_amass_dataset_groups()
        assert isinstance(groups, dict)

    def test_known_groups_present(self):
        groups = get_amass_dataset_groups()
        expected_keys = [
            "AMASS_LOCOMOTION_DATASETS",
            "AMASS_RANDOM_TRAINING_MOTIONS",
            "KIT_KINESIS_TRAINING_MOTIONS",
            "AMASS_BIMANUAL_TRAIN_MOTIONS",
            "AMASS_BIMANUAL_TEST_MOTIONS",
        ]
        for key in expected_keys:
            assert key in groups, f"Expected group '{key}' not found"

    def test_groups_contain_nonempty_lists(self):
        groups = get_amass_dataset_groups()
        for name, paths in groups.items():
            assert isinstance(paths, list), f"{name} should be a list"
            assert len(paths) > 0, f"{name} should not be empty"

    def test_cached_returns_same_object(self):
        a = get_amass_dataset_groups()
        b = get_amass_dataset_groups()
        assert a is b
