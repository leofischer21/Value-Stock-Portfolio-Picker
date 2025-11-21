# dashboard_final.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Root-Verzeichnis bestimmen (für App in app/)
ROOT_DIR = Path(__file__).parent.parent

# Add scripts to path for imports
sys.path.insert(0, str(ROOT_DIR / "scripts"))

# ------------------ Page Config ------------------
st.set_page_config(page_title="Value Stock Picks", layout="wide")
st.title("Value Stock Picks Portfolio")
st.markdown("##### Systematisches Screening nach Value, Quality & Community-Signal")

# ------------------ Sidebar: User-Eingaben ------------------
st.sidebar.header("Portfolio-Einstellungen")

portfolio_size = st.sidebar.selectbox(
    "Portfolio-Größe",
    [5, 10, 20],
    index=2,
    help="Anzahl der Aktien im Portfolio"
)

horizon = st.sidebar.selectbox(
    "Anlagehorizont",
    ["1 Jahr", "2 Jahre", "5+ Jahre"],
    index=2,
    help="Zeithorizont für die Anlage - beeinflusst die Gewichtung von Value vs. Quality"
)

# ------------------ Portfolio berechnen ------------------
@st.cache_data
def calculate_portfolio(portfolio_size: int, horizon: str):
    """Berechnet Portfolio aus monatlichen Daten (mit Caching)"""
    try:
        from portfolio_calculator import calculate_portfolio_from_monthly_data
        portfolio = calculate_portfolio_from_monthly_data(
            portfolio_size=portfolio_size,
            horizon=horizon,
            min_market_cap=30_000_000_000
        )
        return portfolio
    except Exception as e:
        st.error(f"Fehler beim Berechnen des Portfolios: {e}")
        return None

# Berechne Portfolio
df = calculate_portfolio(portfolio_size, horizon)

if df is None or len(df) == 0:
    st.error("Kein Portfolio konnte berechnet werden. Bitte stelle sicher, dass monatliche Daten vorhanden sind.")
    st.info("Führe `python scripts/monthly_update.py` aus, um Daten zu generieren.")
    st.stop()

# MarketCap in Mrd formatieren
if 'marketCap' in df.columns:
    df['Marktkap_Mrd'] = (df['marketCap'] / 1e9).round(1)
else:
    df['Marktkap_Mrd'] = 0.0

# ------------------ Key Metrics ------------------
col1, col2, col3 = st.columns(3)
with col1:
    beta = df['beta'].mean() if 'beta' in df.columns else 0.0
    st.metric("Portfolio-Beta", f"{beta:.2f}")
with col2:
    fwd_pe = df['forwardPE'].mean() if 'forwardPE' in df.columns else 0.0
    st.metric("Forward P/E (gew.)", f"{fwd_pe:.1f}")
with col3:
    comm_score = df['community_score'].mean() if 'community_score' in df.columns else 0.0
    st.metric("Community Score (Ø)", f"{comm_score:.2f}")

st.markdown("---")

# ------------------ Haupttabelle ------------------
st.subheader(f"Top {len(df)} Aktien – Anlagehorizont: {horizon}")

# Schöne Spalten + Runden
display_df = df[['ticker', 'sector', 'Marktkap_Mrd', 'forwardPE',
                 'superinvestor_score', 'reddit_score', 'x_score',
                 'final_score', 'weight_%']].copy()

display_df.columns = ['Ticker', 'Sektor', 'Marktkap (Mrd €)', 'Fwd P/E',
                      'Superinvestoren', 'Reddit', 'X', 'Final Score', 'Gewicht']

# Rundung – clean & einfach
for col in ['Fwd P/E', 'Superinvestoren', 'Reddit', 'X', 'Final Score']:
    if col in display_df.columns:
        display_df[col] = display_df[col].round(2)
if 'Gewicht' in display_df.columns:
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
    if 'weight' in df.columns and 'sector' in df.columns:
        sector_data = df.groupby('sector')['weight'].sum().sort_values(ascending=False)
        fig_pie = px.pie(
            values=sector_data.values,
            names=sector_data.index,
            color_discrete_sequence=px.colors.sequential.Tealgrn
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Sektor-Daten nicht verfügbar")

with colB:
    st.subheader("Community vs. Final Score")
    if 'community_score' in df.columns and 'final_score' in df.columns:
        fig_scatter = px.scatter(
            df,
            x='community_score',
            y='final_score',
            size='weight' if 'weight' in df.columns else None,
            color='sector' if 'sector' in df.columns else None,
            hover_name='ticker',
            size_max=50,
            labels={"community_score": "Community Score", "final_score": "Final Score"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Score-Daten nicht verfügbar")

# ------------------ Download Button ------------------
st.markdown("---")
csv_data = df.to_csv(index=False)
st.download_button(
    label="Portfolio als CSV herunterladen",
    data=csv_data,
    file_name=f"value_portfolio_{portfolio_size}_{horizon.replace('+', 'plus').replace(' ', '_')}.csv",
    mime="text/csv"
)

# ------------------ Info ------------------
st.caption(f"Portfolio-Größe: {portfolio_size} | Anlagehorizont: {horizon} | Berechnet aus monatlichen Daten")
