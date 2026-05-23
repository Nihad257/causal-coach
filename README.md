# 📈 CausalCoach – Did your marketing campaign really work?

**CausalCoach** is a free, open‑source web application that estimates the causal impact of a campaign or intervention using a simple interrupted time series model. 

👉 **Live demo:** [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://causal-coach-kqpvibjvpliqkpqjqdrxaq.streamlit.app/)

## 🎯 What it does

- You upload a CSV with a **date** column and an **outcome** column (e.g., sales, conversions, website visits).
- You select the **start date** of your campaign.
- CausalCoach builds a counterfactual forecast: *what would have happened without the campaign?*
- It then shows you the **estimated causal effect**, a 95% confidence interval, and the probability that the effect is positive.

## 📂 Input data format

Your CSV must have **at least two columns**:

| date       | y     |
|------------|-------|
| 2024-01-01 | 105   |
| 2024-01-08 | 108   |
| ...        | ...   |

- `date` – any date format recognised by pandas (e.g., YYYY-MM-DD).
- `y` – the outcome metric (e.g., sales, revenue, clicks).

> **Note:** The column for the outcome must be named exactly `y`. If your file uses a different name (e.g., `sales`), rename it to `y` before uploading.

## 🚀 How to use (public version)

1. Visit the **[live app](https://causal-coach-kqpvibjvpliqkpqjqdrxaq.streamlit.app/)**.
2. Upload your CSV file.
3. Pick the campaign start date.
4. Click **Run Causal Analysis**.
5. View the counterfactual plot, effect metrics, and probability.
6. (Optional) Download a PDF report.

## 💻 Run locally (for developers)

1. Clone the repository:
   ```bash
   git clone https://github.com/Nihad257/causal-coach.git
   cd causal-coach
