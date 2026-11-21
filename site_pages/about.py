import streamlit as st

st.set_page_config(
    layout="wide"
)

st.title("About the Project")

paginator_flex = st.container(horizontal=True,
    horizontal_alignment="distribute",
    vertical_alignment="center",     
    gap="medium")

if paginator_flex.button("References Page"):
    st.switch_page("site_pages/references.py")
if paginator_flex.button("Methodology Page"):
    st.switch_page("site_pages/methodology.py")