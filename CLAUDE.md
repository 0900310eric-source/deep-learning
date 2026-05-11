# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a personal ML/DL research monorepo with four independent sub-projects. There is no shared build system, package manager, or test framework. Each sub-project is self-contained.

---

## Sub-projects

### `basic_dl/` — Deep Learning Coursework (NumPy + PyTorch)

Homework assignments in plain Python. Each script has a `main()` guard and is run directly.

```bash
# Run from within the hw directory (data files use relative paths)
cd basic_dl/hw1
python HW1_314513050_energy_efficiency.py
python HW1_314513050_ionosphere_data.py

cd basic_dl/hw2
python HW2_314513050_EX1.py   # MNIST CNN
python HW2_314513050_EX2.py   # CIFAR-10 CNN

cd basic_dl/hw3
python HW3_314513050_RNN.py   # Character-level RNN on Shakespeare
```

**Architecture pattern:**
- `hw1`: Fully manual backprop in NumPy — weight matrices stored as instance attributes, explicit `forward_propagation` / `backward_propagation` / `update` methods. No autograd.
- `hw2–hw3`: PyTorch `nn.Module` subclasses. Hyperparameters (lr, epochs, batch_size, kernel sizes, L2 alpha) are module-level constants at the top of each file — change them there, not inside functions.
- All scripts produce `matplotlib` plots via `plt.show()`; there is no file output.

---

### `Resnet_18/` and `Resnet_18_RF/` — ResNet-18 Signal Classification

Jupyter notebooks for two distinct classification problems:

| Project | Dataset | Task |
|---|---|---|
| `Resnet_18/` | RadioML2016.10a (`.pkl`) | Radio modulation recognition |
| `Resnet_18_RF/` | Raw IQ binary (`.bin`) + LTE/WiFi/DVB-T | Wireless technology recognition |

```bash
jupyter notebook Resnet_18/src/test_v3.ipynb
jupyter notebook Resnet_18_RF/src/test_v1.ipynb
```

**Important:** Notebooks contain hardcoded Windows absolute paths (e.g., `C:\Users\eric0\Desktop\...`) for the dataset files, which are not included in the repo (`data/README.txt` notes files were removed due to size). Update these paths before running.

`Resnet_18_RF/` uses `scipy.signal.spectrogram` to convert raw I/Q data into time-frequency representations before feeding to ResNet-18. Saved model checkpoints are `*.pt` files in `src/`.

---

### `test_printer_RAG/` — RAG-based Data Plotting Tool (4 versions)

The most complex project. It is a data visualization tool where signal-processing logic (NumPy transformations) is retrieved or generated at runtime by an LLM/RAG system. Each version (`v1`–`v4`) is a Python package (`__init__.py` present) that improves the retrieval strategy.

**Run as a module from the repo root** (required for relative imports to resolve):
```bash
python -m test_printer_RAG.v2.test_printer_v2
python -m test_printer_RAG.v3.test_printer_v3_semantic_transformer
python -m test_printer_RAG.v4.test_printer_v4_funcrag
```

**External dependencies required at runtime:**
- v2, v4: `ollama` Python client with models pulled locally:
  - `phi3` (used as router in v2)
  - `qwen2.5-coder:7b` (used for code generation)
- v3: `sentence_transformers` with model `multi-qa-mpnet-base-dot-v1` (~420 MB, downloads on first use)

**Architecture — how the versions relate:**

Each version has a `TestPrinter` class (data loading/plotting), a `ColumnMapping` class (maps user column names → generic `x`/`y`), an `Operator` class (numeric transforms), and a `FunctionRAG` class (retrieves/generates numpy logic). The config JSON drives data loading.

| Version | Retrieval strategy | Router |
|---|---|---|
| v1 | None (operator stubs only) | — |
| v2 | LLM intent routing (phi3) → template prompt → qwen2.5-coder generates `def logic(x, y)` | phi3 via ollama |
| v3 | Semantic embedding cosine similarity (`SentenceTransformer`) against codebook built from JSON descriptions+examples | No LLM router; threshold=0.6 |
| v4 | Keyword weight scoring (keywords +0.1, negatives −1.0) against `status: "verified"` entries only | No LLM router |

**RAG library JSON schema evolution:**
- v2 (`numpy_lib_printer_v2.json`): `intent`, `description` (string), `examples`, `code`, `bad_cases`
- v4 (`numpy_lib_printer_v4.json`): `function_id`, `status` ("verified"/"draft"), `keywords` (list), `negatives` (list), `description` (list of strings), `code`

All generated functions must follow this exact signature: `def logic(x, y):` — `x` is the input array, `y` is optional parameter (scalar, dict, or None). Functions must use NumPy vectorization only (no Python loops).

**Config JSON structure (`config_printer_v*.json`):**
```json
{
  "settings": { "output_dir", "input_dir", "raw_dir", "file_extension", "figure_specs": { ... } },
  "mapping": { "group": {...}, "label": {...}, "level": {...} }
}
```
Data files are organized as `group_label_level.csv`. The v2+ config adds `mapping_mask` and `mapping_filter` for selective loading.

---

### `offline_dash_exp/` — MPEG-DASH Streaming Experiment

Pure browser experiment — no Python logic. Use `player_v2.html` (not `player.html`).

```bash
# Serve from the parent of offline_dash_exp/
python -m http.server 8000
# Then open: http://<server-ip>:8000/offline_dash_exp/player_v2.html
```

Video categories available as DASH streams: `animation1/`, `movie1/`, `music1/`, `news1/`, `sport1/`. Switch between them by changing the MPD path input in the player UI (e.g. `/movie1/stream.mpd`). The player logs application-layer stats (bitrate, buffer health, timestamps) downloadable as CSV.

The experimental setup is: Windows HTTP server → browser client → second host running `tc` (Linux traffic control) for bandwidth shaping + Wireshark PCAP capture.

---

## Dependencies (no requirements.txt exists)

| Sub-project | Key packages |
|---|---|
| `basic_dl` | `numpy`, `pandas`, `matplotlib`, `torch`, `torchvision` |
| `Resnet_18*` | `torch`, `torchvision`, `numpy`, `matplotlib`, `scipy` |
| `test_printer_RAG` | `pandas`, `numpy`, `matplotlib`, `torch` (v3), `sentence_transformers` (v3), `ollama` (v2/v4) |
| `offline_dash_exp` | Browser only (Shaka Player via CDN) |

CUDA is used automatically when available (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).
