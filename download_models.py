"""Download pinned OmniJev source and weights, or verify an existing local copy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import urllib.request
import zipfile


SOURCE_REVISION = "14dbec4f71e194852c8d7b88ab36ef639493f400"
SOURCE_URL = f"https://codeload.github.com/tinnel123666888/OmniJev/zip/{SOURCE_REVISION}"
SOURCE_DIRECTORY = "vendor/OmniJev"
SOURCE_MARKER = ".omnijev_source.json"
MODEL_SPECS = {
    "adapter": {
        "repo_id": "tinnel123/OmniJev-0.8B",
        "revision": "3c9aa908f3c9d3f3ccf16bb281ee705d9e98252f",
        "directory": "models/OmniJev-0.8B",
        "required": [
            "adapter_config.json", "adapter_model.safetensors", "head.pt",
            "head_meta.json", "new_tok_emb.pt", "ord.pt", "processor_config.json",
            "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
            "release_manifest.json", "LICENSE", "README.md",
        ],
    },
    "base": {
        "repo_id": "Qwen/Qwen3.5-0.8B",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "directory": "models/Qwen3.5-0.8B",
        "required": [
            "config.json", "model.safetensors-00001-of-00001.safetensors",
            "model.safetensors.index.json", "preprocessor_config.json",
            "video_preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
            "merges.txt", "vocab.json", "chat_template.jinja", "LICENSE", "README.md",
        ],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(directory: Path, names: list[str]) -> list[dict]:
    records = []
    for name in sorted(names):
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing or unexpected file: {path}")
        records.append({"path": name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    return records


def _check_inventory(directory: Path, records: list[dict], verify_hashes: bool) -> bool:
    if not records:
        return False
    for record in records:
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            return False
        path = directory / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size != record["size_bytes"]:
            return False
        if verify_hashes and _sha256(path) != record["sha256"]:
            return False
    return True


def _write_json(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(data, handle, indent=2)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_source(root: Path, local_files_only: bool) -> dict:
    destination = root / SOURCE_DIRECTORY
    marker_path = destination / SOURCE_MARKER
    if destination.is_symlink():
        raise RuntimeError(f"Refusing to use a symlink as the source directory: {destination}")
    if destination.exists():
        if not marker_path.is_file():
            raise RuntimeError(f"Refusing to overwrite an unmarked source directory: {destination}")
        marker = json.loads(marker_path.read_text())
        if marker.get("revision") != SOURCE_REVISION:
            raise RuntimeError(f"Source revision differs from the pinned revision: {destination}")
        if not all((destination / name).is_file() for name in ("mso/infer.py", "LICENSE")):
            raise RuntimeError(f"Required source files are missing: {destination}")
        if not _check_inventory(destination, marker.get("files", []), True):
            raise RuntimeError(f"Source files have changed or are incomplete: {destination}")
        return marker
    if local_files_only:
        raise RuntimeError(f"Pinned source is unavailable offline: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".omnijev-source-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        archive = temporary / "source.zip"
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "ESE6800-model-setup"})
        print("Downloading pinned OmniJev source...", flush=True)
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        expected_prefix = f"OmniJev-{SOURCE_REVISION}"
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                name = PurePosixPath(member.filename)
                if (name.is_absolute() or ".." in name.parts or not name.parts
                        or name.parts[0] != expected_prefix or "\\" in member.filename
                        or stat.S_ISLNK(member.external_attr >> 16)):
                    raise RuntimeError(f"Unsafe source archive member: {member.filename}")
            bundle.extractall(temporary / "extracted")
        extracted = temporary / "extracted" / expected_prefix
        for name in ("mso/infer.py", "LICENSE"):
            if not (extracted / name).is_file():
                raise RuntimeError(f"Source archive is missing {name}")
        names = [path.relative_to(extracted).as_posix() for path in extracted.rglob("*") if path.is_file()]
        marker = {
            "revision": SOURCE_REVISION, "url": SOURCE_URL,
            "archive_sha256": _sha256(archive), "directory": SOURCE_DIRECTORY,
            "files": _inventory(extracted, names),
        }
        _write_json(extracted / SOURCE_MARKER, marker)
        extracted.rename(destination)
    return marker


def _validate_adapter_manifest(directory: Path, records: list[dict]) -> None:
    release = json.loads((directory / "release_manifest.json").read_text())
    if release.get("release") != "v1.1" or release.get("repo_id") != MODEL_SPECS["adapter"]["repo_id"]:
        raise RuntimeError("Unexpected OmniJev release manifest")
    for record in records:
        if record["path"] == "release_manifest.json":
            continue
        expected = release["files"].get(record["path"])
        if expected is None or (expected["bytes"], expected["sha256"]) != (record["size_bytes"], record["sha256"]):
            raise RuntimeError(f"OmniJev release checksum mismatch: {record['path']}")


def prepare_assets(
    project_root: str | Path, *, verify_hashes: bool = False, local_files_only: bool = False
) -> dict:
    """Return a verified asset manifest; download only missing pinned assets.

    Artifact ``directory`` values are relative to project_root; inventory ``path``
    values are relative to that directory. Initial downloads always get SHA-256
    inventories. Reuse checks sizes, and also hashes when verify_hashes=True.
    local_files_only=True prevents all network requests.
    """
    root = Path(project_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "model_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    source = _ensure_source(root, local_files_only)
    models = {}
    for key, spec in MODEL_SPECS.items():
        directory = root / spec["directory"]
        old = previous.get("models", {}).get(key, {})
        identifiers_match = all(old.get(field) == spec[field] for field in ("repo_id", "revision", "directory"))
        files = old.get("files", [])
        required_present = {entry["path"] for entry in files} == set(spec["required"])
        reusable = identifiers_match and required_present and _check_inventory(directory, files, verify_hashes)
        if not reusable:
            if local_files_only:
                raise RuntimeError(f"Offline assets are missing, changed, or not recorded at the pinned revision: {directory}")
            from huggingface_hub import hf_hub_download, snapshot_download

            print(f"Preparing {spec['repo_id']} at {spec['revision']}...", flush=True)
            if identifiers_match:
                for record in files:
                    path = directory / record["path"]
                    if path.exists() and not _check_inventory(directory, [record], verify_hashes):
                        if record["path"] not in spec["required"]:
                            raise RuntimeError("Unexpected path in the previous model manifest")
                        hf_hub_download(
                            repo_id=spec["repo_id"], filename=record["path"],
                            revision=spec["revision"], local_dir=str(directory),
                            token=False, force_download=True,
                        )
            snapshot_download(
                repo_id=spec["repo_id"], revision=spec["revision"],
                local_dir=str(directory), allow_patterns=spec["required"],
                token=False,
            )
            files = _inventory(directory, spec["required"])
            if key == "adapter":
                _validate_adapter_manifest(directory, files)
        models[key] = {field: spec[field] for field in ("repo_id", "revision", "directory")}
        models[key]["files"] = files
    manifest = {
        "schema_version": 1, "project_root": str(root),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "source": source, "models": models,
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--verify-hashes", action="store_true", help="Hash existing weights as well as checking sizes")
    parser.add_argument("--offline", action="store_true", help="Use local files only; never access the network")
    args = parser.parse_args()
    prepare_assets(args.project_root, verify_hashes=args.verify_hashes, local_files_only=args.offline)
    print("Model files ready.", flush=True)


if __name__ == "__main__":
    main()
