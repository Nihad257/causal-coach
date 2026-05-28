# 📈 CausalCoach – Did your marketing campaign really work?

**CausalCoach** is a free, open-source web application that estimates the causal impact of a campaign or intervention using a simple interrupted time series model.

---

## 🎯 What it does

- You upload a CSV with a **date** column and an **outcome** column (e.g., sales, conversions, website visits).
- You select the **start date** of your campaign.
- CausalCoach builds a counterfactual forecast: *what would have happened without the campaign?*
- It then shows you the **estimated causal effect**, a 95% confidence interval, and the probability that the effect is positive.

---

## 📂 Input data format

Your CSV must have **at least two columns**:

| date       | y   |
|------------|-----|
| 2024-01-01 | 105 |
| 2024-01-08 | 108 |
| ...        | ... |

- `date` – any date format recognised by pandas (e.g., YYYY-MM-DD).
- `y` – the outcome metric (e.g., sales, revenue, clicks).

> **Note:** The outcome column must be named exactly `y`.  
> If your file uses a different name (e.g., `sales`), rename it to `y` before uploading.

---

## 🚀 How to use (public version)

1. Visit https://causal-coach.streamlit.app/
3. Upload your CSV file.
4. Pick the campaign start date.
5. Click **Run Causal Analysis**.
6. View the counterfactual plot, effect metrics, and probability.


---

## 💻 Run locally

### 1. Clone the repository

```bash
git clone https://github.com/Nihad257/causal-coach.git
cd causal-coach
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

### 4. Open in browser

```text
http://localhost:8501
```

---

## 🧠 How it works

The tool fits a linear regression model to the pre-campaign period:

```text
y = β₀ + β₁·time + β₂·post + β₃·time_post + ε
```

Where:

- `time` – a simple time trend
- `post` – indicator for post-campaign weeks
- `time_post` – trend change after the campaign

The counterfactual is what would have happened if `post` and `time_post` remained zero.

The gap between actual and counterfactual after the start date is the estimated causal effect.

Uncertainty is quantified with:

- 95% confidence intervals from the model
- One-sample t-test on the effect series

---

## 📦 Dependencies

- Python 3.9+
- streamlit
- pandas
- numpy
- statsmodels
- plotly
- scipy


See `requirements.txt` for exact versions.

---

## 📄 License

MIT – free to use, modify, and share.

---

## 🙋 Feedback & contributions

Open an issue or pull request on GitHub.
