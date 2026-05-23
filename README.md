# CausalCoach

A pure-Python Streamlit app that runs a **Bayesian structural time series**
causal impact analysis on a time series you upload. No R required.

Built on [`tfcausalimpact`](https://github.com/WillianFuks/tfcausalimpact),
a TensorFlow Probability port of Google's `CausalImpact` R package.

## Features

- Upload a CSV with `date` (YYYY-MM-DD) and `y` (numeric) columns.
- Optionally include extra numeric columns as covariates (control series).
- Pick the campaign start date with a date picker.
- Validation: ≥8 pre-campaign and ≥3 post-campaign observations.
- Outputs: average effect, 95% credible interval, relative lift, P(effect > 0).
- Interactive Plotly chart of actual vs. counterfactual with shaded CI.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>.

## Deploy to Streamlit Cloud (free, one-click public URL)

1. Push this folder to a public GitHub repo.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **New app**, pick your repo, branch, and `app.py` as the entrypoint.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically.

You'll get a public URL like `https://<your-app>.streamlit.app`.

> First boot may take 1–2 minutes because TensorFlow Probability is large.

## CSV format

```csv
date,y
2024-01-01,1023
2024-01-08,1098
...
```

Extra columns are treated as optional covariates. A `sample_data.csv` with a
known ~20% lift starting 2024-07-01 is included.

## Notes

- `tfcausalimpact` uses TensorFlow Probability under the hood. The first
  inference call compiles the model and can take ~30s; subsequent runs are
  faster.
- All file paths are relative; no environment variables required.
