"""(round 13, I18) Image tag policy: the committed kustomization names a deploy-time placeholder, never a commit hash
that goes stale on the next commit; CI and the kind smoke test set their own tags (docs/deployment.md, "Image tags")."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _images(path: str) -> dict[str, str]:
    kustomization = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    return {image["name"]: str(image["newTag"]) for image in kustomization.get("images") or []}


def test_the_committed_kustomization_names_the_deploy_time_placeholder():
    assert _images("deploy/k8s/kustomization.yaml") == {"finsight": "set-at-deploy"}
    # never a commit hash: it would name an older build as soon as anything else is committed
    assert not re.fullmatch(r"[0-9a-f]{7,40}", _images("deploy/k8s/kustomization.yaml")["finsight"])


def test_the_smoke_overlay_sets_its_own_tag():
    assert _images("deploy/k8s-smoke/kustomization.yaml") == {"finsight": "smoke"}


def test_ci_checks_the_placeholder_and_renders_the_commit_tag():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "image: finsight:set-at-deploy$" in workflow
    assert 'newTag: \\"${GITHUB_SHA::7}\\"' in workflow
    assert 'grep -Eq "finsight:${GITHUB_SHA::7}$"' in workflow


def test_the_policy_is_documented_in_both_languages():
    for path in ("docs/deployment.md", "docs/zh/deployment.md"):
        text = (ROOT / path).read_text(encoding="utf-8")
        assert "set-at-deploy" in text, path
        assert "kustomize edit set image finsight=finsight:$TAG" in text, path
