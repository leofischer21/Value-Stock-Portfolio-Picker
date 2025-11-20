# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import glob

# Root-Verzeichnis bestimmen (für App in app/)
ROOT_DIR = Path(__file__).parent.parent

# ------------------ Page Config ------------------
st.set_page_config(page_title="Value Stock Picks", layout="wide")
st.title("Value Stock Picks Portfolio")
st.markdown("##### Systematisches Screening nach Value, Quality & Community-Signal")

# ------------------ Daten laden ------------------
csv_files = sorted(glob.glob(str(ROOT_DIR / "examples/portfolio_*.csv")), reverse=True)
if not csv_files:
    st.error("Keine Portfolio-CSVs in /examples/ gefunden!")
    st.stop()

latest_file = csv_files[0]
df = pd.read_csv(latest_file)

# MarketCap in Mrd
df['Marktkap_Mrd'] = (df['marketCap'] / 1e9).round(1)

# ------------------ Sidebar ------------------
st.sidebar.header("Portfolio auswählen")
selected_file = st.sidebar.selectbox("Version", csv_files, format_func=lambda x: Path(x).stem.replace("portfolio_", ""))
if selected_file != latest_file:
    df = pd.read_csv(selected_file)

# ------------------ Key Metrics ------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Portfolio-Beta", f"{df['beta'].mean():.2f}")
with col2:
    st.metric("Forward P/E (gew.)", f"{df['forwardPE'].mean():.1f}")
with col3:
    st.metric("Community Score (Ø)", f"{df['community_score'].mean():.2f}")

st.markdown("---")

# ------------------ Haupttabelle ------------------
st.subheader(f"Aktuelle Auswahl – {Path(selected_file).stem.replace('portfolio_', '')}")

# Schöne Spalten + Runden
display_df = df[['ticker', 'sector', 'Marktkap_Mrd', 'forwardPE',
                 'superinvestor_score', 'reddit_score', 'x_score',
                 'final_score', 'weight_%']].copy()

display_df.columns = ['Ticker', 'Sektor', 'Marktkap (Mrd €)', 'Fwd P/E',
                      'Superinvestoren', 'Reddit', 'X', 'Final Score', 'Gewicht']

# Rundung – clean & einfach
for col in ['Fwd P/E', 'Superinvestoren', 'Reddit', 'X', 'Final Score']:
    display_df[col] = display_df[col].round(2)
display_df['Gewicht'] = display_df['Gewicht'].str.replace('%', ' %')

st.dataframe(
    display_df.style.background_gradient(cmap='Blues', subset=['Final Score']),
    use_container_width=True,
    hide_index=True
)

# ------------------ Charts ------------------
colA, colB = st.columns(2)

with colA:
    st.subheader("Gewichtung nach Sektor")
    sector_data = df.groupby('sector')['weight'].sum().sort_values(ascending=False)
    fig_pie = px.pie(
        values=sector_data.values,
        names=sector_data.index,
        color_discrete_sequence=px.colors.sequential.Tealgrn
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with colB:
    st.subheader("Community vs. Final Score")
    fig_scatter = px.scatter(
        df,
        x='community_score',
        y='final_score',
        size='weight',
        color='sector',
        hover_name='ticker',
        size_max=50,
        labels={"community_score": "Community Score", "final_score": "Final Score"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------ Historie (auskommentiert – später mit Monatsfilter) ------------------
# st.markdown("---")
# st.subheader("Historische Top-10 Entwicklung")
# st.info("Kommt später – nur monatlich/quartalsweise")

st.caption(f"Daten vom: {Path(selected_file).stem.replace('portfolio_', '')}")