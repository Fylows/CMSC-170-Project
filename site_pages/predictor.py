import streamlit as st

paginator_flex = st.container(horizontal=True,
    horizontal_alignment="distribute",
    vertical_alignment="center",     
    gap="medium")

if paginator_flex.button("Discussion Page"):
    st.switch_page("site_pages/discussion.py")
if paginator_flex.button("References Page"):
    st.switch_page("site_pages/references.py")