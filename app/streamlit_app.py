"""CCARC Road Damage Research - Interactive Demo and Results Explorer (Streamlit Application)."""

import streamlit as st


def main():
    st.set_page_config(
        page_title="CCARC Road Damage Research Explorer",
        page_icon="🛣️",
        layout="wide",
    )
    st.title("🛣️ CCARC: Cross-Country Road Damage Detection")
    st.markdown(
        """
### Country- and Class-Aware Reliability Consistency Learning for Cross-Country Road Damage Detection

This application is an interactive research explorer for:
- Qualitative detection comparisons across RDD2022 country domains.
- Leave-One-Country-Out (LOCO) performance benchmark analysis.
- Consistency reliability maps and class-level diagnostic metrics.

*Status: Environment and project scaffolding initialized.*
"""
    )


if __name__ == "__main__":
    main()
