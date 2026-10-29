"""Tracked update automation is configuration, not a prose-to-constant comparison."""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _dependabot():
    return yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text())


def _assert_dependabot(config, files):
    assert files == [".github/dependabot.yml"], "unexpected or missing GitHub configuration"
    assert config["version"] == 2
    updates = config["updates"]
    assert len(updates) == 1, "one dependency update job owns the project"
    update = updates[0]
    assert update["package-ecosystem"] == "uv", "ecosystem must update uv.lock"
    assert update["directory"] == "/", "update job must cover the project root"
    assert update["schedule"]["interval"] == "weekly", "update cadence must be weekly"
    groups = update["groups"]
    assert len(groups) == 1, "dependency updates must form one wildcard group"
    group = next(iter(groups.values()))
    assert group["patterns"] == ["*"], "group must cover every dependency"
    assert not group.get("exclude-patterns"), "group excludes dependencies"
    assert "dependency-type" not in group, "group restricts dependency types"


def test_the_only_github_file_is_weekly_grouped_root_uv_automation():
    done = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", ".github"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    _assert_dependabot(_dependabot(), sorted(set(done.stdout.rstrip("\0").split("\0"))))


@pytest.mark.parametrize(
    "plant", ["cadence", "ecosystem", "root", "ungrouped", "two-groups", "exclusion", "workflow"]
)
def test_update_automation_lock_rejects_policy_drift(plant):
    config = copy.deepcopy(_dependabot())
    update = config["updates"][0]
    files = [".github/dependabot.yml"]
    if plant == "cadence":
        update["schedule"]["interval"] = "monthly"
        reason = "cadence"
    elif plant == "ecosystem":
        update["package-ecosystem"] = "pip"
        reason = "ecosystem"
    elif plant == "root":
        update["directory"] = "/subproject"
        reason = "project root"
    elif plant == "ungrouped":
        next(iter(update["groups"].values()))["patterns"] = ["numpy"]
        reason = "every dependency"
    elif plant == "two-groups":
        update["groups"]["other"] = {"patterns": ["*"]}
        reason = "one wildcard group"
    elif plant == "exclusion":
        next(iter(update["groups"].values()))["exclude-patterns"] = ["numpy"]
        reason = "excludes dependencies"
    else:
        files.append(".github/workflows/ci.yml")
        reason = "GitHub configuration"
    with pytest.raises(AssertionError, match=reason):
        _assert_dependabot(config, files)
