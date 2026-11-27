import streamlit as st

st.title("References")
st.write("This section lists the sources used in the making of this project.")

# References: ___________________________________________________________________________

st.divider()

st.header("Sources")

col_dataset_source, col_reg_sources = st.columns(2)
col_metric_sources, col_frontend_sources = st.columns(2)

with col_dataset_source.container(border = True):
    st.subheader("Dataset")

    st.write("Department of Energy. (2023). *Power Generation by Fuel Source, 1990-2020* [Data set]. https://data.gov.ph/index/public/dataset/Power%20Generation%20by%20Fuel%20Source,%201990-2020/mhb14dc0-8tio-af0u-6nac-euxrn6bz48rm")

with col_reg_sources.container(border = True):
    st.subheader("Regression Models")

    st.write("GeeksforGeeks. (2025a). *Implementation of polynomial regression*. GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/python-implementation-of-polynomial-regression/")
    st.write("GeeksforGeeks. (2025b). *Ridge regression*. GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/what-is-ridge-regression/")
    st.write("GeeksforGeeks. (2026a). *Linear Regression in Machine learning*. GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/")
    st.write("GeeksforGeeks. (2026b). *Implementation of Ridge Regression from Scratch using Python*. GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/implementation-of-ridge-regression-from-scratch-using-python/")

with col_metric_sources.container(border = True):
    st.subheader("Evaluation Metrics")

    st.write("Wikipedia contributors. (2026a). *Mean absolute error*. Wikipedia. https://en.wikipedia.org/wiki/Mean_absolute_error")
    st.write("Wikipedia contributors. (2026b). *Mean absolute percentage error*. Wikipedia. https://en.wikipedia.org/wiki/Mean_absolute_percentage_error")
    st.write("Wikipedia contributors. (2026c). *Root mean square deviation*. Wikipedia. https://en.wikipedia.org/wiki/Root_mean_square_deviation")

with col_frontend_sources.container(border = True):
    st.subheader("Frontend")

    st.write("*Create an app - Streamlit Docs*. (n.d.). https://docs.streamlit.io/get-started/tutorials/create-an-app")
    st.write("*st.pyplot - Streamlit docs*. (n.d.). https://docs.streamlit.io/develop/api-reference/charts/st.pyplot")
    st.write("*Streamlit API cheat sheet - Streamlit Docs*. (n.d.). https://docs.streamlit.io/develop/quick-reference/cheat-sheet")

# AI Declaration: _______________________________________________________________________

st.divider()

st.header("AI Declaration")

with st.container(border = True):
    st.subheader("Notice of AI Tools Usage")

    st.write("**AI tools** have been used in this project for the purpose of **debugging** the regression models and **ensuring consistency** " \
    "within the user interface (alongside the cited Streamlit cheatsheet above). The content has been **reviewed** to ensure that the final" \
    "output is **original** and as **accurate** as possible.")

# _______________________________________________________________________________________

paginator_flex = st.container(horizontal=True,
    horizontal_alignment="distribute",
    vertical_alignment="center",     
    gap="medium")

if paginator_flex.button("Predictor Page"):
    st.switch_page("site_pages/predictor.py")
if paginator_flex.button("About Page"):
    st.switch_page("site_pages/about.py")