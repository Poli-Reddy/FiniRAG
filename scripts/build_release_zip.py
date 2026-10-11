"""Build a clean source-only release archive (audit P0-1/P0-3, #26, #17-19).

The working tree legitimately contains things that must never be
distributed: local secrets (.env and every .env.* variant), VCS metadata
(.git), caches, the evaluation dataset, extracted financial results, and
prior reports. This script produces FiniRAG-<version>.zip with source,
migrations, tests, frontend, docs and packaging files only — and refuses
to run if a secret-bearing file would be included.

Usage:
    python scripts/build_release_zip.py [--version X.Y.Z] [--out dist/]
        [--include-eval-artifacts] [--allow-unscannable]

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
# Exact filenames that are never packaged.
EXCLUDE_FILES = set()
# .env and every .env.* variant are excluded by default; only explicitly
# allowlisted redacted templates (currently just .env.example) may ship
# (audit finding #17).
ENV_FILE_RE = re.compile(r"^\.env(\..+)?$")
ENV_TEMPLATE_ALLOWLIST = {".env.example"}
# Extracted financial results + generated reports are data, not source.
# Excluded by default; --include-eval-artifacts opts back in for a PRIVATE
# data-recovery package (audit finding #18).
EVAL_ARTIFACT_RES = [
    re.compile(r"^evaluation/results/"),
    re.compile(r"^evaluation/rag_live_report\.json$"),
    re.compile(r"^evaluation/rag_report\.json$"),
]
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".log", ".tmp", ".bak", ".sqlite", ".db")
# Extensions known to be binary: packaged with an explicit warning list,
# never secret-scanned byte-wise.
KNOWN_BINARY_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".ico",
                         ".woff", ".woff2", ".pptx", ".tag"}
# Any other extension that fails UTF-8 decoding refuses the release
# unless --allow-unscannable is passed (fail closed, finding #17).

# Secret-shaped content that must not appear in any packaged text file.
# Known-fake test fixtures are exempted with surgical precision below —
# never by weakening the patterns themselves.
SECRET_PATTERNS = [
    re.compile(r"gsk_[A-Za-z0-9]{8,}"),  # Groq
    re.compile(r"AIza[0-9A-Za-z_-]{10,}"),  # Gemini
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    # Any credentialed URL, including postgresql+asyncpg:// (finding #17):
    # scheme://user:pass@host — the fixture allowlist below exempts
    # documentation hosts only.
    re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^/\s]+:[^/\s@]+@[^/\s]+"),
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
    # For credentialed URLs, inspect the host that follows the "@".
    m = re.search(r"@([^/:]+)", candidate)
    return bool(m and m.group(1).lower() in FIXTURE_HOSTS)


def _included_files(include_eval_artifacts: bool = False) -> list[Path]:
    files: list[Path] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        posix = rel.as_posix()
        if ENV_FILE_RE.match(rel.name) and rel.name not in ENV_TEMPLATE_ALLOWLIST:
            continue
        if rel.name in EXCLUDE_FILES or path.suffix in EXCLUDE_SUFFIXES:
            continue
        if not include_eval_artifacts and any(rx.search(posix) for rx in EVAL_ARTIFACT_RES):
            continue
        files.append(path)
    return files


def _scan_for_secrets(files: list[Path],
                      allow_unscannable: bool = False,
                      ) -> tuple[list[str], list[str]]:
    """Returns (offenders, binary_packaged).

    Refuse the release if any packaged text file looks secret-bearing
    (finding #17). Files that cannot be decoded as UTF-8 refuse the
    release unless their extension is a known binary (packaged with a
    warning) or --allow-unscannable is passed.
    """
    offenders: list[str] = []
    binary_packaged: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            try:
                rel = path.relative_to(ROOT).as_posix()
            except ValueError:
                rel = str(path)
            if path.suffix.lower() in KNOWN_BINARY_SUFFIXES:
                binary_packaged.append(rel)
                continue
            if allow_unscannable:
                binary_packaged.append(f"{rel} (UNSCANNED --allow-unscannable)")
                continue
            offenders.append(f"{rel}: unreadable-as-text (refusing; "
                             "pass --allow-unscannable to override explicitly)")
            continue
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                if _is_fixture(text, match):
                    continue
                try:
                    shown = path.relative_to(ROOT)
                except ValueError:
                    shown = path
                offenders.append(f"{shown}: {match.group(0)[:24]}…")
                break
            else:
                continue
            break
    return offenders, binary_packaged


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="2.0.0")
    parser.add_argument("--out", default="dist")
    parser.add_argument("--include-eval-artifacts", action="store_true",
                        help="ALSO package evaluation/results/ + generated "
                             "reports. These contain EXTRACTED FINANCIAL DATA "
                             "— use only for a private data-recovery package, "
                             "never for a public source release.")
    parser.add_argument("--allow-unscannable", action="store_true",
                        help="Package files that cannot be secret-scanned. "
                             "Explicit opt-in only.")
    args = parser.parse_args(argv)

    files = _included_files(include_eval_artifacts=args.include_eval_artifacts)
    offenders, binary_packaged = _scan_for_secrets(
        files, allow_unscannable=args.allow_unscannable)
    if offenders:
        print("REFUSED: secret-shaped content in release files:", file=sys.stderr)
        for o in offenders:
            print(f"  - {o}", file=sys.stderr)
        raise SystemExit(1)
    if not any(str(f.relative_to(ROOT)) == ".env.example" for f in files):
        raise SystemExit("REFUSED: .env.example (redacted template) missing from release")
    if args.include_eval_artifacts:
        print("WARNING: packaging evaluation/results/ + generated reports — "
              "these contain EXTRACTED FINANCIAL DATA. Private use only.",
              file=sys.stderr)
    for b in binary_packaged:
        print(f"NOTE: packaging unscanned binary: {b}", file=sys.stderr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"FiniRAG-{args.version}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(ROOT))
    print(f"wrote {archive} ({len(files)} files, no .env*/.git/data)")


if __name__ == "__main__":
    main()
