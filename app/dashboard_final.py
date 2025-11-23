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
@st.cache_data(ttl=3600)  # Cache für 1 Stunde
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

# Warnung außerhalb der gecachten Funktion
if len(df) < portfolio_size:
    st.warning(f"⚠️ Nur {len(df)} Aktien gefunden (angefordert: {portfolio_size}). Möglicherweise nicht genug qualifizierte Aktien verfügbar.")

# MarketCap in Mrd formatieren
if 'marketCap' in df.columns:
    df['Marktkap_Mrd'] = (df['marketCap'] / 1e9).round(1)
else:
    df['Marktkap_Mrd'] = 0.0

# ------------------ Key Metrics ------------------
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    # Calculate weighted portfolio beta
    if 'beta' in df.columns and 'weight' in df.columns:
        portfolio_beta = (df['beta'].fillna(1.0) * df['weight']).sum()
    else:
        portfolio_beta = df['beta'].mean() if 'beta' in df.columns else 0.0
    st.metric("Portfolio-Beta", f"{portfolio_beta:.2f}")
with col2:
    # Gewichteter Forward P/E (nur positive Werte)
    if 'forwardPE' in df.columns and 'weight' in df.columns:
        positive_fwd_pe = df[(df['forwardPE'] > 0) & (df['forwardPE'].notna())]
        if len(positive_fwd_pe) > 0:
            fwd_pe = (positive_fwd_pe['forwardPE'] * positive_fwd_pe['weight']).sum() / positive_fwd_pe['weight'].sum()
        else:
            fwd_pe = 0.0
    elif 'forwardPE' in df.columns:
        positive_fwd_pe = df[(df['forwardPE'] > 0) & (df['forwardPE'].notna())]
        fwd_pe = positive_fwd_pe['forwardPE'].mean() if len(positive_fwd_pe) > 0 else 0.0
    else:
        fwd_pe = 0.0
    st.metric("Forward P/E (gew.)", f"{fwd_pe:.1f}")
with col3:
    comm_score = df['community_score'].mean() if 'community_score' in df.columns else 0.0
    st.metric("Community Score (Ø)", f"{comm_score:.2f}")
with col4:
    if 'priceMomentum12M' in df.columns:
        avg_momentum = df['priceMomentum12M'].dropna().mean()
        if pd.notna(avg_momentum):
            st.metric("12M Momentum (Ø)", f"{avg_momentum:.1f}%")
        else:
            st.metric("12M Momentum (Ø)", "N/A")
    else:
        st.metric("12M Momentum (Ø)", "N/A")
with col5:
    # Portfolio Predicted Performance (based on selected horizon)
    horizon_to_cagr = {
        "1 Jahr": "predicted_cagr_1y",
        "2 Jahre": "predicted_cagr_2y",
        "5+ Jahre": "predicted_cagr_5y"
    }
    cagr_col = horizon_to_cagr.get(horizon, "predicted_cagr_2y")
    if cagr_col in df.columns and 'weight' in df.columns:
        # Calculate weighted average, handling NaN values
        valid_mask = df[cagr_col].notna() & df['weight'].notna()
        if valid_mask.any():
            portfolio_predicted = (df.loc[valid_mask, cagr_col] * df.loc[valid_mask, 'weight']).sum()
            if pd.notna(portfolio_predicted):
                horizon_label = horizon.replace("+", "+")
                st.metric(f"Pred. Performance ({horizon_label})", f"{portfolio_predicted:.1f}%")
            else:
                st.metric("Pred. Performance", "N/A")
        else:
            st.metric("Pred. Performance", "N/A")
    else:
        st.metric("Pred. Performance", "N/A")
with col6:
    # Average AI Moat Score
    if 'ai_moat_score' in df.columns and 'weight' in df.columns:
        valid_mask = df['ai_moat_score'].notna() & df['weight'].notna()
        if valid_mask.any():
            avg_ai_moat = (df.loc[valid_mask, 'ai_moat_score'] * df.loc[valid_mask, 'weight']).sum()
            st.metric("AI Moat Score (gew.)", f"{avg_ai_moat:.2f}")
        else:
            st.metric("AI Moat Score", "N/A")
    elif 'ai_moat_score' in df.columns:
        avg_ai_moat = df['ai_moat_score'].mean()
        st.metric("AI Moat Score (Ø)", f"{avg_ai_moat:.2f}")
    else:
        st.metric("AI Moat Score", "N/A")

st.markdown("---")

# ------------------ Haupttabelle ------------------
st.subheader(f"Top {len(df)} Aktien – Anlagehorizont: {horizon}")

# Schöne Spalten + Runden
display_cols = ['ticker', 'sector', 'Marktkap_Mrd', 'forwardPE',
                'reddit_score', 'x_score',
                'final_score', 'weight_%']
if 'priceMomentum12M' in df.columns:
    display_cols.insert(-2, 'priceMomentum12M')  # Insert before final_score

# Add AI scores if available
if 'ai_moat_score' in df.columns:
    display_cols.insert(-2, 'ai_moat_score')  # Insert before final_score
if 'ai_quality_score' in df.columns:
    display_cols.insert(-2, 'ai_quality_score')  # Insert before final_score

# Add predicted performance column based on horizon
horizon_to_cagr = {
    "1 Jahr": "predicted_cagr_1y",
    "2 Jahre": "predicted_cagr_2y",
    "5+ Jahre": "predicted_cagr_5y"
}
cagr_col = horizon_to_cagr.get(horizon, "predicted_cagr_2y")
if cagr_col in df.columns:
    display_cols.insert(-2, cagr_col)  # Insert before final_score

display_df = df[display_cols].copy()

col_names = ['Ticker', 'Sektor', 'Marktkap (Mrd €)', 'Fwd P/E',
             'Reddit', 'X', 'Final Score', 'Gewicht']
if 'priceMomentum12M' in display_df.columns:
    col_names.insert(-2, '12M Momentum')  # Insert before Final Score
if 'ai_moat_score' in display_df.columns:
    col_names.insert(-2, 'AI Moat')  # Insert before Final Score
if 'ai_quality_score' in display_df.columns:
    col_names.insert(-2, 'AI Quality')  # Insert before Final Score
if cagr_col in display_df.columns:
    horizon_label = horizon.replace("+", "+")
    col_names.insert(-2, f'Pred. Perf. ({horizon_label})')  # Insert before Final Score

display_df.columns = col_names

# Format 12M Momentum as percentage
if '12M Momentum' in display_df.columns:
    display_df['12M Momentum'] = display_df['12M Momentum'].apply(
        lambda x: f"{float(x):.1f}%" if pd.notna(x) else "N/A"
    )

# Format Predicted Performance as percentage
pred_perf_col = f'Pred. Perf. ({horizon.replace("+", "+")})'
if pred_perf_col in display_df.columns:
    display_df[pred_perf_col] = display_df[pred_perf_col].apply(
        lambda x: f"+{float(x):.1f}%" if pd.notna(x) and float(x) > 0 else f"{float(x):.1f}%" if pd.notna(x) else "N/A"
    )

# Rundung – clean & einfach
for col in ['Fwd P/E', 'Reddit', 'X', 'AI Moat', 'AI Quality', 'Final Score']:
    if col in display_df.columns:
        display_df[col] = display_df[col].round(2)
if 'Gewicht' in display_df.columns:
    display_df['Gewicht'] = display_df['Gewicht'].str.replace('%', ' %')

# Try to use background gradient if matplotlib is available, otherwise plain dataframe
try:
    st.dataframe(
        display_df.style.background_gradient(cmap='Blues', subset=['Final Score']),
        width='stretch',
        hide_index=True
    )
except ImportError:
    # Fallback if matplotlib not available
    st.dataframe(
        display_df,
        width='stretch',
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
        st.plotly_chart(fig_pie, width='stretch')
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
        st.plotly_chart(fig_scatter, width='stretch')
    else:
        st.info("Score-Daten nicht verfügbar")

# ------------------ Erweiterte Visualisierungen ------------------
st.markdown("---")
st.subheader("Erweiterte Analysen")

# Row 1: Value vs Quality & Score Distribution
col1, col2 = st.columns(2)

with col1:
    st.subheader("Value vs. Quality Score")
    if 'value_score' in df.columns and 'quality_score' in df.columns:
        fig_value_quality = px.scatter(
            df,
            x='value_score',
            y='quality_score',
            size='final_score' if 'final_score' in df.columns else None,
            color='sector' if 'sector' in df.columns else None,
            hover_name='ticker',
            size_max=50,
            labels={"value_score": "Value Score", "quality_score": "Quality Score"},
            title="Trade-off zwischen Value und Quality"
        )
        st.plotly_chart(fig_value_quality, width='stretch')
    else:
        st.info("Value/Quality Score-Daten nicht verfügbar")

with col2:
    st.subheader("Score-Verteilung")
    if 'final_score' in df.columns:
        fig_hist = px.histogram(
            df,
            x='final_score',
            nbins=20,
            labels={"final_score": "Final Score", "count": "Anzahl"},
            title="Verteilung der Final Scores"
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, width='stretch')
    else:
        st.info("Final Score-Daten nicht verfügbar")

# Row 2: Sector Heatmap & Beta Distribution
col3, col4 = st.columns(2)

with col3:
    st.subheader("Sektor-Heatmap")
    if 'sector' in df.columns and all(col in df.columns for col in ['value_score', 'quality_score', 'community_score', 'final_score']):
        # Create heatmap data
        sector_scores = df.groupby('sector').agg({
            'value_score': 'mean',
            'quality_score': 'mean',
            'community_score': 'mean',
            'final_score': 'mean'
        }).round(2)
        
        fig_heatmap = px.imshow(
            sector_scores.T,
            labels=dict(x="Sektor", y="Score-Typ", color="Durchschnitt"),
            x=sector_scores.index,
            y=['Value', 'Quality', 'Community', 'Final'],
            color_continuous_scale='RdYlGn',
            aspect="auto"
        )
        fig_heatmap.update_layout(title="Durchschnittliche Scores pro Sektor")
        st.plotly_chart(fig_heatmap, width='stretch')
    else:
        st.info("Sektor- oder Score-Daten nicht verfügbar")

with col4:
    st.subheader("Beta-Verteilung")
    if 'beta' in df.columns and 'final_score' in df.columns:
        fig_beta = px.scatter(
            df,
            x='beta',
            y='final_score',
            size='weight' if 'weight' in df.columns else None,
            color='sector' if 'sector' in df.columns else None,
            hover_name='ticker',
            size_max=50,
            labels={"beta": "Beta", "final_score": "Final Score"},
            title="Beta vs. Final Score"
        )
        # Add portfolio beta line
        if 'beta' in df.columns and 'weight' in df.columns:
            portfolio_beta_val = (df['beta'].fillna(1.0) * df['weight']).sum()
            fig_beta.add_hline(
                y=df['final_score'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Portfolio Beta: {portfolio_beta_val:.2f}"
            )
        st.plotly_chart(fig_beta, width='stretch')
    else:
        st.info("Beta- oder Score-Daten nicht verfügbar")

# Row 3: Momentum vs Final Score & Predicted Performance vs Final Score
col5, col6 = st.columns(2)

with col5:
    if 'priceMomentum12M' in df.columns and 'final_score' in df.columns:
        st.subheader("12M Momentum vs. Final Score")
        fig_momentum = px.scatter(
            df,
            x='priceMomentum12M',
            y='final_score',
            size='weight' if 'weight' in df.columns else None,
            color='sector' if 'sector' in df.columns else None,
            hover_name='ticker',
            size_max=50,
            labels={"priceMomentum12M": "12M Momentum (%)", "final_score": "Final Score"},
            title="12M Price Momentum vs. Final Score"
        )
        st.plotly_chart(fig_momentum, width='stretch')
    else:
        st.info("Momentum-Daten nicht verfügbar")

with col6:
    if cagr_col in df.columns and 'final_score' in df.columns:
        st.subheader("Predicted Performance vs. Final Score")
        fig_pred = px.scatter(
            df,
            x=cagr_col,
            y='final_score',
            size='weight' if 'weight' in df.columns else None,
            color='sector' if 'sector' in df.columns else None,
            hover_name='ticker',
            size_max=50,
            labels={cagr_col: f"Predicted CAGR ({horizon}) (%)", "final_score": "Final Score"},
            title=f"Predicted Performance vs. Final Score ({horizon})"
        )
        st.plotly_chart(fig_pred, width='stretch')
    else:
        st.info("Predicted Performance-Daten nicht verfügbar")

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
