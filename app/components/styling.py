"""Shared CSS + theming for all pages."""
import streamlit as st

def apply_base_style():
    """Inject minimal global CSS — muted, editorial, FT-ish."""
    st.markdown("""
<style>
/* Tighten default Streamlit padding */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Headings: tighter tracking */
h1, h2, h3 {
    letter-spacing: -0.02em;
    font-weight: 700;
}

/* Metric label uppercase */
[data-testid="stMetricLabel"] {
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.75rem !important;
    color: #64748b;
    font-weight: 600;
}

/* Metric value: mono font */
[data-testid="stMetricValue"] {
    font-family: ui-monospace, 'SF Mono', 'Monaco', 'Menlo', monospace;
    font-weight: 600;
    color: #0c4a6e;
}

/* Sidebar: subtle tint */
[data-testid="stSidebar"] {
    background-color: #f8fafc;
}

/* Buttons: less bouncy */
.stButton > button {
    border-radius: 0.375rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    transition: all 0.15s ease;
}

/* Caption in a lighter tone */
[data-testid="stCaptionContainer"] {
    color: #64748b;
}

/* Horizontal rule: less prominent */
hr {
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
    border-color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)
