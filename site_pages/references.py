import streamlit as st

st.title("References")
st.write("This section lists the sources used in the making of this project.")

# References: ___________________________________________________________________________

st.write("*Create an app - Streamlit Docs*. (n.d.). https://docs.streamlit.io/get-started/tutorials/create-an-app")
st.write("GeeksforGeeks. (2025a). *Implementation of polynomial regression*. GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/python-implementation-of-polynomial-regression/")
st.write("GeeksforGeeks. (2025b). *Ridge regression*. GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/what-is-ridge-regression/")
st.write("GeeksforGeeks. (2026a). *Linear Regression in Machine learning*. GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/")
st.write("GeeksforGeeks. (2026b). *Implementation of Ridge Regression from Scratch using Python*. GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/implementation-of-ridge-regression-from-scratch-using-python/")
st.write("*st.pyplot - Streamlit docs*. (n.d.). https://docs.streamlit.io/develop/api-reference/charts/st.pyplot")
st.write("*Streamlit API cheat sheet - Streamlit Docs*. (n.d.). https://docs.streamlit.io/develop/quick-reference/cheat-sheet")
st.write("Wikipedia contributors. (2026a). *Mean absolute error*. Wikipedia. https://en.wikipedia.org/wiki/Mean_absolute_error")
st.write("Wikipedia contributors. (2026b). *Mean absolute percentage error*. Wikipedia. https://en.wikipedia.org/wiki/Mean_absolute_percentage_error")
st.write("Wikipedia contributors. (2026c). *Root mean square deviation*. Wikipedia. https://en.wikipedia.org/wiki/Root_mean_square_deviation")

# _______________________________________________________________________________________

paginator_flex = st.container(horizontal=True,
    horizontal_alignment="distribute",
    vertical_alignment="center",     
    gap="medium")

if paginator_flex.button("Predictor Page"):
    st.switch_page("site_pages/predictor.py")
if paginator_flex.button("About Page"):
    st.switch_page("site_pages/about.py")