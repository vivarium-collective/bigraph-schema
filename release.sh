#!/usr/bin/env bash
# Bump version, commit, tag, and push. CI publishes to PyPI via OIDC
# trusted publisher (.github/workflows/release.yml).
#
# Usage:
#   ./release.sh           # bumps patch
#   ./release.sh minor     # bumps minor
#   ./release.sh major     # bumps major

set -e

bump_kind="${1:-patch}"

# Working tree must be clean.
if [ -n "$(git status --untracked-files=no --porcelain)" ]; then
    echo "You have uncommitted changes. Aborting."
    exit 1
fi

# Must be on main, up to date with origin.
branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$branch" != "main" ]; then
    echo "You are on $branch but releases must be cut from main."
    exit 1
fi
git fetch origin main
if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
    echo "main is not in sync with origin/main. Pull or push first."
    exit 1
fi

# Bump version, refresh lock so the local-package entry tracks it.
uv version --bump "$bump_kind"
version=$(uv version --short)
uv lock

# Commit pyproject + lock together.
git add pyproject.toml uv.lock
git commit -m "version $version"

# Push the commit first so CI runs against main HEAD.
git push origin main

# Tag the commit and push the tag — this triggers release.yml,
# which re-runs tests and then publishes to PyPI via OIDC.
git tag -m "version v$version" "v$version"
git push origin "v$version"

echo "Tagged v$version. Watch the release workflow:"
echo "  gh run watch"
