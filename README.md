# Decision-aware pruning for OmniJev

Sivangi Chatterjee, Amit Ramasubramanian, Tien Dee Lin

## Which layers can we skip?

Section 14 of [Tien Dee’s executed notebook](Tien_Dee_Layer_Selection_Colab.ipynb) builds on [Amit’s iteration 5](iter5_probe_wanda_colab%282%29.ipynb). We skipped one layer at a time, counted how many practice answers changed from the full model, then tried the top 2, 4, 6 and 8 layers together. Each practice set contains 600 questions.

For the six-layer comparison, the ranking replaced layer 16 with layer 22. When skipping those layers in both stages, AI2D accuracy went from 62.2% to 63.2% and POPE accuracy went from 78.6% to 78.0%. Median time per question went from 152.5 to 149.3 ms on AI2D and from 142.4 to 138.9 ms on POPE. Accuracy used 500 held-out questions per dataset. Timing used 40 questions per dataset on the same T4.

- Find the layer IDs for each count in [chosen_layers.json](experiments/layer_selection/chosen_layers.json).
- Compare [individual layers](experiments/layer_selection/practice_layer_ranking.csv), [group accuracy on practice questions](experiments/layer_selection/practice_layer_groups.csv), and [all five settings on held-out questions](experiments/layer_selection/held_out_results.csv).
- The [full run archive](experiments/layer_selection/Tien_Dee_Layer_Selection_Results.zip) contains the 29,000 per-question predictions, 400 individual timings, original split IDs, model revisions, and verification summary. Both splits came from the public test files. For a larger held-out evaluation, exclude the practice questions and the POPE photos used for practice.

### Reproduce the run in Colab

1. Open the [notebook in Google Colab](https://colab.research.google.com/github/sivangichatterjee/Decision-Aware-Pruning-for-Efficient-Typed-Decision-Models/blob/tien-dee-layer-selection/Tien_Dee_Layer_Selection_Colab.ipynb) and choose a T4 GPU. If Colab cannot open the GitHub link, download the notebook from this branch and use **File → Upload notebook** in Colab. Leave `QUICK = False`. Set `SAVE_TO_DRIVE = True` in §0 if you want checkpoints to survive a VM reset.
2. Run §§0–3 and the **first code cell of §4**. If Colab needs a kernel restart after installing packages, rerun §0, then resume at the imports cell in §1.
3. Jump to **§14** and run its cells in order. The first code cell sets the candidate layers and group sizes. The recorded T4 run took about 94 minutes from the first practice prediction through final timing, plus setup time.

The notebook prints the output directory in §14. The archive’s `run_identity.json` records the exact question IDs for comparison if the upstream datasets change.

## Local OmniJev baseline

This earlier notebook runs the downloaded OmniJev-0.8B model in VS Code.

1. Open `OmniJev_Baseline.ipynb` in VS Code.
2. Use the project **.venv (Python 3.13.0)** kernel at the top right.
3. Click **Run All**. To try your own input, edit `TEXT` in the last cell and run that cell again.

If updating from the earlier notebook, reopen it and restart the kernel once.
The three code cells load the model, classify sample messages, and classify your
own text. The weights and Python packages are already installed on the prepared
Mac. Output uses ordinary tables; no widgets or separate JupyterLab server are
needed.

### Supporting files

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
