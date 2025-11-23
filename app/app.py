# app.py
import streamlit as st
import pandas as pd
from pathlib import Path

# Root-Verzeichnis bestimmen (für App in app/)
ROOT_DIR = Path(__file__).parent.parent

st.set_page_config(page_title="Value Moat Portfolio", layout="wide")
st.title("Value Moat Portfolio Generator")
st.markdown("Monatlich aktualisiert – powered by Superinvestoren, Reddit & X")

# User-Eingaben
col1, col2 = st.columns(2)
with col1:
    size = st.selectbox("Portfolio-Größe", [5, 10, 20], index=2)
with col2:
    horizon = st.selectbox("Anlagehorizont", ["1 Jahr", "2 Jahre", "5+ Jahre"])

# Aktuelles Portfolio laden
latest = sorted((ROOT_DIR / "examples").glob("portfolio_*.csv"))[-1]
df = pd.read_csv(latest)

st.success(f"Aktuellster Stand: {latest.stem.replace('portfolio_', '')}")

# Tabelle anzeigen
display = df.head(size)[['ticker','sector','forwardPE','superinvestor_score','reddit_score','x_score','ki_moat_score','final_score','weight_%']].round(3)
st.dataframe(display, use_container_width=True)

st.download_button("Portfolio herunterladen", data=df.head(size).to_csv(index=False), file_name=f"my_value_portfolio_{size}.csv")