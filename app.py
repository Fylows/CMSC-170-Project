import matplotlib
matplotlib.use("Agg")   # Figures show in actual page.

import streamlit as st
from main import Linear, Polynomial, Ridge, LoadData, PlotComparison

# RunModels(): __________________________________________________________________________
@st.cache_data    # For training models only once.
def RunModels():
    X_train, X_test, Y_train, Y_test, future_years, fuel_sources = LoadData()

    # Run regression models.
    linear_results = Linear(X_train, Y_train, X_test, Y_test, fuel_sources, future_years)
    polynomial_results = Polynomial(X_train, Y_train, X_test, Y_test, fuel_sources, future_years, degree=4)
    ridge_results = Ridge(X_train, Y_train, X_test, Y_test, fuel_sources, future_years)

    # Put models' results in dictionary.
    results = {"Linear": linear_results, 
               "Polynomial": polynomial_results, 
               "Ridge": ridge_results}

    return results, X_test, Y_test, future_years, fuel_sources



# __________________________________________________________________________
# Streamlit
# Ref: https://docs.streamlit.io/get-started/tutorials/create-an-app
# ---- https://docs.streamlit.io/develop/quick-reference/cheat-sheet 
# ---- https://docs.streamlit.io/develop/api-reference/charts/st.pyplot 

st.title("Power Generation")
st.write("Comparison of Regression Models")

with st.spinner(text="Running regression models..."):
    results, X_test, Y_test, future_years, fuel_sources = RunModels()

# optionss
model_name = st.selectbox("Choose model", ["Linear", "Polynomial", "Ridge"])
selected_model = results[model_name]

st.subheader(model_name + "Regression Metrics")

colA, colB, colC = st.columns(3)
# Print errors in columns
with colA:
    st.metric("MAE", f"{selected_model['overall_metrics']['mae']:.3f}")

with colB:
    st.metric("MAPE", f"{selected_model['overall_metrics']['mape']:.3f}%")

with colC:
    st.metric("RMSE", f"{selected_model['overall_metrics']['rmse']:.3f}")


st.subheader(model_name + "Regression Metrics")

# from main.py. Plot Comaprison uses matplot lib to graph the regression res
fig = PlotComparison(
    results["Linear"]["overall_metrics"],
    results["Polynomial"]["overall_metrics"],
    results["Ridge"]["overall_metrics"]
)

# show figure
st.pyplot(fig)
