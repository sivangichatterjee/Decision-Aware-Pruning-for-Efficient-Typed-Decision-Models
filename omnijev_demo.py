"""Run the real OmniJev checkpoint locally and preserve its probability outputs.

Upstream code and weights are downloaded by download_models.py. A small runtime
overlay only adds device/dtype constructor arguments to the pinned upstream
loader. The original downloaded source and all learned weights remain intact.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time


LOADER_EDITS = (
    (
        "def __init__(self, ckpt, base_model, tiny=False, max_pixels=768 * 28 * 28, align_options=True):",
        "def __init__(self, ckpt, base_model, tiny=False, max_pixels=768 * 28 * 28, align_options=True, device=None, dtype=None):",
    ),
    (
        'self.dev = torch.device("cpu" if tiny else acc)',
        'self.dev = torch.device(device or ("cpu" if tiny else acc))',
    ),
    (
        'self.dtype = torch.float32 if self.dev.type == "cpu" else torch.bfloat16',
        'self.dtype = dtype if dtype is not None else (torch.float32 if self.dev.type == "cpu" else torch.bfloat16)',
    ),
)


def _runtime_source(project_root: Path, manifest: dict) -> Path:
    """Create a disposable copy with three explicit loader edits, never weights."""
    source = project_root / manifest["source"]["directory"]
    original = (source / "mso" / "infer.py").read_text()
    patched = original
    for before, after in LOADER_EDITS:
        if patched.count(before) != 1:
            raise RuntimeError("Pinned OmniJev loader changed; inspect the runtime patch before continuing.")
        patched = patched.replace(before, after, 1)
    digest = hashlib.sha256(patched.encode()).hexdigest()
    overlay = project_root / ".cache" / "omnijev_runtime" / digest
    target = overlay / "mso" / "infer.py"
    if not target.exists():
        overlay.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source / "mso", overlay / "mso", dirs_exist_ok=True)
        target.write_text(patched)
        shutil.copy2(source / "LICENSE", overlay / "LICENSE")
    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Runtime loader copy was changed. Remove its .cache directory and rerun.")
    return overlay


def load_model(project_root: str | Path, device: str = "auto"):
    """Load downloaded weights offline. Auto uses CUDA if present, else CPU.

    Apple MPS is deliberately not selected: this checkpoint's full MPS path has
    not been validated. CUDA without native BF16 support uses FP32 for this
    initial correctness demo, rather than silently using unsupported BF16.
    """
    from download_models import prepare_assets

    # Model-loading progress bars otherwise request browser widget JavaScript.
    # Set this here so VS Code works without a special launch environment.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()

    project_root = Path(project_root).expanduser().resolve()
    manifest = prepare_assets(project_root, local_files_only=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["MSO_FLA"] = "0"  # optional CUDA/Triton kernels are not needed
    os.environ["MSO_PANELS"] = str(project_root / ".cache" / "panels")
    import huggingface_hub.constants
    huggingface_hub.constants.HF_HUB_OFFLINE = True
    import torch

    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Choose device='auto', 'cpu', or 'cuda'.")
    selected = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this runtime has no available CUDA GPU.")
    dtype = torch.float32
    if selected == "cuda":
        try:
            native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            native_bf16 = torch.cuda.get_device_capability()[0] >= 8
        if native_bf16:
            dtype = torch.bfloat16
    overlay = _runtime_source(project_root, manifest)
    if "mso.infer" in sys.modules:
        existing = Path(sys.modules["mso.infer"].__file__).resolve()
        if existing != overlay / "mso" / "infer.py":
            raise RuntimeError("A different OmniJev source is already imported. Restart the notebook kernel.")
    sys.path.insert(0, str(overlay))
    MSO1 = importlib.import_module("mso.infer").MSO1
    start = time.perf_counter()
    model = MSO1(
        str(project_root / manifest["models"]["adapter"]["directory"]),
        str(project_root / manifest["models"]["base"]["directory"]),
        device=selected,
        dtype=dtype,
    )
    model.model.eval()
    model._demo_root = project_root
    model._demo_runtime = {
        "device": selected,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "transformers_version": importlib.import_module("transformers").__version__,
        "peft_version": importlib.import_module("peft").__version__,
        "load_seconds": round(time.perf_counter() - start, 4),
        "source_revision": manifest["source"]["revision"],
        "adapter_revision": manifest["models"]["adapter"]["revision"],
        "base_revision": manifest["models"]["base"]["revision"],
        "loader_overlay_sha256": overlay.name,
        "inference": "local PyTorch; model loading is offline",
    }
    if selected == "cuda":
        model._demo_runtime["gpu"] = torch.cuda.get_device_name(0)
    print(f"Loaded real OmniJev-0.8B on {selected} using {dtype} in {model._demo_runtime['load_seconds']:.1f}s.")
    return model


def classify_text(
    model,
    text: str,
    labels: Mapping[str, str],
    statement: str | None = None,
    levels: Sequence[str] | None = None,
    score_question: str | None = None,
    image_path: str | Path | None = None,
) -> dict:
    """Classify text using the real model, with explicit blank visual context.

    An optional image_path replaces the blank image for image-and-text examples.
    labels maps each allowed category name to its natural-language description.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Provide nonempty text to classify.")
    if not isinstance(labels, Mapping) or len(labels) < 2:
        raise ValueError("Provide at least two category names and descriptions.")
    if any(not isinstance(k, str) or not k.strip() or k == "[abstain]" for k in labels):
        raise ValueError("Category names must be nonempty strings; '[abstain]' is reserved.")
    if any(not isinstance(v, str) for v in labels.values()):
        raise ValueError("Category descriptions must be strings.")
    if (levels is None) != (score_question is None):
        raise ValueError("Provide levels and score_question together, or omit both.")
    if levels is not None and (len(levels) < 2 or len(set(levels)) != len(levels)):
        raise ValueError("Use at least two distinct ordered score levels.")

    from PIL import Image
    import torch

    if image_path is None:
        path = model._demo_root / ".cache" / "blank_context.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            Image.new("RGB", (112, 112), color="white").save(path)
        context_note = "The image is a blank placeholder. Use only the supplied text."
    else:
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        context_note = "Use the supplied image and text."
    # Upstream consumes images + question instructions; state['text'] is ignored.
    context = f"{context_note}\n\nText to assess:\n{text}\n\n"
    questions = {
        "category": {
            "type": "choice",
            "instructions": context + "Which category best describes this input?",
            "criteria": dict(labels),
        }
    }
    if statement is not None:
        questions["statement"] = {"type": "noul", "instructions": context + statement}
    if levels is not None:
        questions["score"] = {
            "type": "score", "instructions": context + score_question, "levels": list(levels)
        }
    if model.dev.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        answers = model.system_one({"images": [str(path)]}, questions)
    if model.dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    run = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "visual_input": "blank placeholder" if image_path is None else str(path),
        "questions": questions,
        "answers": answers,
        "elapsed_seconds": round(elapsed, 4),
        "timing_note": "Single demonstration call, not a warmed repeated benchmark.",
        "runtime": dict(model._demo_runtime),
    }
    # Verify outputs; never replace an unexpected model result with a demo answer.
    table = summarize_answers(run)
    for question, rows in table.groupby("question"):
        probabilities = rows["probability"].tolist()
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
            raise RuntimeError(f"Invalid probability returned for {question}: {answers[question]}")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=0.01):
            raise RuntimeError(f"Probability mass is incomplete for {question}: {answers[question]}")
    return run


def summarize_answers(run: dict):
    """Readable probability table including abstention and the noul complement."""
    import pandas as pd

    rows = []
    for question, answer in run["answers"].items():
        kind = run["questions"][question]["type"]
        if kind == "noul":
            p = float(answer["noul"])
            options = {"true": p, "false": 1.0 - p}
        else:
            options = dict(answer["probabilities"])
            if kind == "choice":
                options["[abstain]"] = answer["abstain"]
        for option, probability in options.items():
            rows.append({"question": question, "type": kind, "option": option, "probability": float(probability)})
    return pd.DataFrame(rows, columns=["question", "type", "option", "probability"])


def save_run(run: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return path
