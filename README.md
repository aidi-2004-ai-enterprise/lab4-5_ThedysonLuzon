# AIDI-2004 — Labs 4 & 5 (Thedyson Luzon)

**Exploratory Data Analysis (Lab 4)** and a **minimal training pipeline (Lab 5)** for a binary classification task.  
This repo includes an HTML export of the EDA, a Jupyter notebook, a lightweight Python training script, and a demo GIF.

<p align="center">
  <img src="aidi2004lab5.gif" alt="Lab 4–5 demo" width="720">
</p>

---

## What’s in this repo

- `lab4.ipynb` — Interactive EDA notebook (Lab 4).
- `lab4.html` — One-click, shareable HTML version of the EDA.
- `lab4_decisions.md` — Written rationale for data prep & modeling choices. 
- `training_pipeline.py` — Script to train/evaluate a model (Lab 5).
- `requirements.txt` — Python dependencies to reproduce results.
- `data/raw/` — Raw dataset files used.
- `reports/` — Place to save figures/metrics/exports from runs.
- `aidi2004lab5.gif` — Demo animation embedded above.

---

## Quickstart

> Requires **Python 3.10+**.

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Put csv into data/raw/

# 4) Run the training pipeline (Lab 5)
python training_pipeline.py

# 5) View the EDA (Lab 4)
#   a) Open the interactive notebook:
#      jupyter lab  # then open lab4.ipynb
#   b) Or open the static export in your browser:
#      open lab4.html   # Windows: start lab4.html
```