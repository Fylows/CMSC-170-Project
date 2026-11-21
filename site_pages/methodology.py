import streamlit as st

paginator_flex = st.container(horizontal=True,
    horizontal_alignment="distribute",
    vertical_alignment="center",     
    gap="medium")

if paginator_flex.button("About Page"):
    st.switch_page("site_pages/about.py")
if paginator_flex.button("Results Page"):
    st.switch_page("site_pages/results.py")