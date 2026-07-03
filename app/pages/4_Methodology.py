"""Methodology — the audit-standard model documentation, rendered from docs/."""

import streamlit as st

from riskplatform.config import PROJECT_ROOT

st.set_page_config(page_title="Methodology", page_icon="📉", layout="wide")

doc = PROJECT_ROOT / "docs" / "methodology.md"
if doc.exists():
    st.markdown(doc.read_text())
else:
    st.error("docs/methodology.md not found in the deployment.")
