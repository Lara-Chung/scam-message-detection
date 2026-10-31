# Scam Message Detection

Robust scam message detection across SMS, email and phone dialogue.

## Setup

```
conda create -n scam-detection python=3.11 -y && conda run -n scam-detection pip install -r requirements.txt
```

Using venv instead also works: create an environment with Python 3.11 and run `pip install -r requirements.txt`.

## Layout

| Path | Content |
|---|---|
| `data/raw/` | Downloaded datasets (not committed) |
| `data/processed/` | Cleaned CSVs and `splits.csv` |
| `src/` | Shared modules |
| `notebooks/` | One notebook per experiment (E1 to E6) and `00_data_summary.ipynb` |
| `tests/` | pytest tests |
| `app/` | Streamlit app |

## Data

Download instructions: TBD.
