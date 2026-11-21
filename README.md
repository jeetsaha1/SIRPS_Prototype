# SIRPS Prototype

SIRPS is a mock-first, synthetic ESG-aware 63-trading-day return research prototype. It is not a real-market data source or investment advisory system.

## Requirements

- Python 3.11
- Git
- VS Code (recommended)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python data_generation/make_company_master.py
python data_generation/make_mock_data.py
pytest -q tests/
streamlit run app.py
```

## Development page runner

```bash
streamlit run dev_run.py -- home
streamlit run dev_run.py -- model_comparison
```

## Mock mode

The application automatically uses `data/mock/` when a real generated file is absent.

PowerShell:

```powershell
$env:SIRPS_USE_MOCK="1"
streamlit run app.py
```

The demo banner identifies mock usage. Final demo releases must not load files from `data/mock/`.

## Data rules

- `SEED = 42`
- Entry: close of `t+1`
- Exit: close of `t+64`
- Features at date `t` may use only information available by `t`
- Every CSV ends with `data_source = SYNTHETIC_DEMO`
- Synthetic data must never be presented as real market evidence
