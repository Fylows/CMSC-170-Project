import streamlit as st


st.set_page_config(page_title="CMSC 170 Project", page_icon="🤖", layout="wide")

pg = st.navigation([
    st.Page("site_pages/about.py",   title="About"),
    st.Page("site_pages/methodology.py", title="Methodology"),
    st.Page("site_pages/results.py",  title="Results"),
    st.Page("site_pages/discussion.py",  title="Discussion"),
    st.Page("site_pages/predictor.py",  title="Predictor"),
    st.Page("site_pages/references.py",  title="References"),

])
pg.run()