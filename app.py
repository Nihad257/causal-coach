import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from scipy.stats import ttest_1samp

st.set_page_config(page_title="CausalCoach", layout="wide")
st.title("📈 CausalCoach – Did your campaign really work?")

uploaded_file = st.file_uploader("Upload CSV (must have columns: date, y)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True, errors='coerce')
df = df.dropna(subset=['date'])
    st.dataframe(df.head())
    
    campaign_date = st.date_input("Campaign start date",
                                  min_value=df['date'].min(),
                                  max_value=df['date'].max())
    
    if st.button("Run Causal Analysis"):
        df = df.sort_values('date')
        df['time'] = np.arange(len(df))
        df['post'] = (df['date'] >= pd.to_datetime(campaign_date)).astype(int)
        df['time_post'] = df['time'] * df['post']
        
        X = df[['time', 'post', 'time_post']]
        X = sm.add_constant(X)
        model = sm.OLS(df['y'], X).fit()
        
        X_counter = X.copy()
        X_counter['post'] = 0
        X_counter['time_post'] = 0
        counterfactual = model.predict(X_counter)
        
        effect = df['y'] - counterfactual
        avg_effect = effect[df['post'] == 1].mean()
        ci = effect[df['post'] == 1].quantile([0.025, 0.975])
        rel_lift = avg_effect / df['y'][df['post'] == 1].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['date'], y=df['y'], mode='lines', name='Actual', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df['date'], y=counterfactual, mode='lines', name='Counterfactual', line=dict(color='blue')))
        pred = model.get_prediction(X)
        pred_summary = pred.summary_frame(alpha=0.05)
        fig.add_trace(go.Scatter(x=df['date'], y=pred_summary['obs_ci_upper'], fill=None, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df['date'], y=pred_summary['obs_ci_lower'], fill='tonexty', mode='lines', line=dict(width=0),
                                 fillcolor='rgba(0,100,80,0.2)', name='95% CI'))
        fig.add_vline(x=campaign_date, line_dash="dash", line_color="green")
        st.plotly_chart(fig)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Average effect", f"{avg_effect:.2f}", delta=f"{rel_lift*100:.1f}%")
        col2.metric("95% CI lower", f"{ci.iloc[0]:.2f}")
        col3.metric("95% CI upper", f"{ci.iloc[1]:.2f}")
        
        t_stat, p_value = ttest_1samp(effect[df['post'] == 1], 0)
        prob_positive = 1 - p_value/2 if t_stat > 0 else p_value/2
        st.success(f"Probability that effect is positive: {prob_positive*100:.1f}%")
