# OmniJev baseline

Sivangi Chatterjee, Amit Ramasubramanian, Tien Dee Lin

One notebook for running the downloaded OmniJev-0.8B model locally in VS Code.

## Use it

1. Open `OmniJev_Baseline.ipynb` in VS Code.
2. Use the project **.venv (Python 3.13.0)** kernel at the top right.
3. Click **Run All**. To try your own input, edit `TEXT` in the last cell and run that cell again.

If updating from the earlier notebook, reopen it and restart the kernel once.
The three code cells load the model, classify sample messages, and classify your
own text. The weights and Python packages are already installed on the prepared
Mac. Output uses ordinary tables; no widgets or separate JupyterLab server are
needed.

## Supporting files

- `omnijev_demo.py`: loads the local model and runs classification.
- `download_models.py`: downloads the pinned model files for a new checkout.
- `requirements.txt`: Python dependencies.
- `model_manifest.json`: exact source/model revisions and file checksums.

For a fresh checkout, run these once from the project folder, then select its
`.venv` kernel in VS Code:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_models.py
```

The actual [OmniJev adapter and decision heads](https://huggingface.co/tinnel123/OmniJev-0.8B)
and [Qwen3.5-0.8B backbone](https://huggingface.co/Qwen/Qwen3.5-0.8B) live in `models/`
(about 1.89 GB). [Upstream source](https://github.com/tinnel123666888/OmniJev) lives
in `vendor/`. Loading and inference use those local files. Large assets and the
Python environment are ignored by Git; the download helper reproduces them.

The helper disables widget progress bars and uses a small loader adjustment for
device/precision selection. Text is supplied in the model's question instructions
with a blank image, as required by the upstream interface. The original downloaded
weights are unchanged. This baseline runs on CPU on the prepared Mac; it is an
inference demo, not an accuracy evaluation or pruning benchmark.
