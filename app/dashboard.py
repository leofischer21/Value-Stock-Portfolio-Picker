# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import glob
import os

# Root-Verzeichnis bestimmen (für App in app/)
ROOT_DIR = Path(__file__).parent.parent

st.set_page_config(page_title="Leo's Value Moat Portfolio", layout="wide")
st.title("Leo's Deep Value + Superinvestor Portfolio")
st.markdown("##### Buffett-style mit Community-Power – Stand 18.11.2025")

# --- Alle Portfolios laden ---
csv_files = sorted(glob.glob(str(ROOT_DIR / "examples/portfolio_*.csv")), reverse=True)
if not csv_files:
    st.error("Keine Portfolio-CSVs in /examples/ gefunden!")
    st.stop()

latest_file = csv_files[0]
df = pd.read_csv(latest_file)

# MarketCap schön machen
df['marketCap_B'] = (df['marketCap'] / 1e9).round(1)

# --- Sidebar Auswahl ---
st.sidebar.header("Portfolio-Version")
selected_file = st.sidebar.selectbox("Wähle einen Run", csv_files, index=0)
if selected_file != latest_file:
    df = pd.read_csv(selected_file)

# --- Haupt-Dashboard ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Portfolio-Beta", f"{df['beta'].mean():.2f}")
with col2:
    st.metric("Durchschn. Forward P/E", f"{df['forwardPE'].mean():.1f}")
with col3:
    st.metric("Superinvestor-Durchschnitt", f"{df['superinvestor_score'].mean():.2f}")
with col4:
    st.metric("Anzahl Runs", len(csv_files))

st.markdown("---")

# Tabelle
st.subheader(f"Top {len(df)} Aktien – {Path(selected_file).stem}")
show_cols = ['ticker','sector','marketCap_B','forwardPE','superinvestor_score',
             'reddit_score','x_score','final_score','weight_%']
display_df = df[show_cols].copy()
display_df.columns = ['Ticker','Sektor','Marktkap (Mrd)','Fwd P/E','Superinvestor','Reddit','X','Final Score','Gewicht']
display_df = display_df.round(3)
st.dataframe(display_df.style.background_gradient(cmap='viridis', subset=['Final Score']), use_container_width=True)

# --- Charts ---
colA, colB = st.columns(2)

with colA:
    st.subheader("Gewichte nach Sektor")
    sector_weights = df.groupby('sector')['weight'].sum().sort_values(ascending=False)
    fig1 = px.pie(values=sector_weights.values, names=sector_weights.index, color_discrete_sequence=px.colors.sequential.Blues_r)
    st.plotly_chart(fig1, use_container_width=True)

with colB:
    st.subheader("Superinvestor-Score vs. Final Score")
    fig2 = px.scatter(df, x='superinvestor_score', y='final_score', size='weight',
                      hover_name='ticker', color='sector', size_max=60,
                      labels={"superinvestor_score": "Superinvestor Score", "final_score": "Final Score"})
    st.plotly_chart(fig2, use_container_width=True)

# Historie
st.markdown("---")
st.subheader("Historische Entwicklung")
if len(csv_files) > 1:
    history = []
    for f in csv_files[:20]:  # nur letzte 20
        tmp = pd.read_csv(f).head(10)  # Top 10
        date = Path(f).stem.split('_')[1] + " " + Path(f).stem.split('_')[2]
        for _, row in tmp.iterrows():
            history.append({"Date": date, "Ticker": row['ticker'], "Rank": tmp.index[tmp['ticker']==row['ticker']].tolist()[0]+1})
    hist_df = pd.DataFrame(history)
    fig3 = px.line(hist_df, x="Date", y="Rank", color="Ticker", markers=True,
                   title="Top-10 Entwicklung über die Zeit (niedriger = besser)")
    fig3.update_yaxes(autorange="reversed")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Noch keine Historie – führe das Skript mehrmals aus")

st.success(f"Dashboard läuft – Daten vom {Path(selected_file).stem}")