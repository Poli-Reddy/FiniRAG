"""Build a clean source-only release archive (audit P0-1/P0-3, #26).

The working tree legitimately contains things that must never be
distributed: local secrets (.env), VCS metadata (.git), caches, the
evaluation dataset, generated outputs, and prior reports. This script
produces FiniRAG-<version>.zip with source, migrations, tests, frontend,
docs and packaging files only — and refuses to run if a secret-bearing
file would be included.

Usage:
    python scripts/build_release_zip.py [--version X.Y.Z] [--out dist/]

After any credential incident, ROTATE first (all provider keys + the
database password), update Render/local envs, and only then cut a release:
an archive is a point-in-time copy and cannot un-share old secrets.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Never distributed, no exceptions. .env holds live provider keys and the
# database password; .git may embed historic secrets; Dataset/, outputs and
# caches are data, not source.
EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv",
                "node_modules", "Dataset", "sample_outputs", ".claude"}
EXCLUDE_FILES = {".env"}
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".log", ".tmp", ".bak", ".sqlite", ".db")

# Secret-shaped content that must not appear in any packaged text file.
# Known-fake test fixtures are exempted with surgical precision below —
# never by weakening the patterns themselves.
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{8,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{10,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"postgres(ql)?://[^/\s]+:[^/\s@]+@"),
]

# Exact documented-fake sentinels used by redaction tests (e.g.
# "AIzaSyBADSECRET must not appear in logs"). Anything else matching a
# pattern above refuses the release.
SENTINEL_ALLOWLIST = {"AIzaSyBADSECRET"}

# URL hosts that prove a connection string is a fixture, not a credential.
FIXTURE_HOSTS = {"host", "localhost", "example.com", "test"}


def _is_fixture(text: str, match: re.Match) -> bool:
    candidate = match.group(0)
    if candidate in SENTINEL_ALLOWLIST:
        return True
    # For connection strings, inspect the host that follows the match —
    # the pattern itself ends at the "@".
    host = re.match(r"([^/:]+)", text[match.end():])
    return bool(host and host.group(1).lower() in FIXTURE_HOSTS)


def _included_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if rel.name in EXCLUDE_FILES or path.suffix in EXCLUDE_SUFFIXES:
            continue
        files.append(path)
    return files


def _scan_for_secrets(files: list[Path]) -> list[str]:
    """Refuse the release if any packaged file looks secret-bearing."""
    offenders: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            continue  # binary/unreadable: not scanned, still packaged
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                if _is_fixture(text, match):
                    continue
                offenders.append(f"{path.relative_to(ROOT)}: {match.group(0)[:24]}…")
                break
            else:
                continue
            break
    return offenders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="2.0.0")
    parser.add_argument("--out", default="dist")
    args = parser.parse_args()

    files = _included_files()
    offenders = _scan_for_secrets(files)
    if offenders:
        print("REFUSED: secret-shaped content in release files:", file=sys.stderr)
        for o in offenders:
            print(f"  - {o}", file=sys.stderr)
        raise SystemExit(1)
    if not any(str(f.relative_to(ROOT)) == ".env.example" for f in files):
        raise SystemExit("REFUSED: .env.example (redacted template) missing from release")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"FiniRAG-{args.version}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(ROOT))
    print(f"wrote {archive} ({len(files)} files, no .env/.git/data)")


if __name__ == "__main__":
    main()
