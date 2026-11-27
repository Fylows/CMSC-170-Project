import streamlit as st

st.title("References")
st.write("This section lists the sources used in the making of this project.")

paginator_flex = st.container(horizontal=True,
    horizontal_alignment="distribute",
    vertical_alignment="center",     
    gap="medium")

if paginator_flex.button("Predictor Page"):
    st.switch_page("site_pages/predictor.py")
if paginator_flex.button("About Page"):
    st.switch_page("site_pages/about.py")