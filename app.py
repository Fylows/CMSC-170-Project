import numpy as np

import matplotlib
matplotlib.use("Agg")   # Figures show in actual page.

import matplotlib.pyplot as plt

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

# _______________________________________________________________________________________

# Streamlit
# Ref: https://docs.streamlit.io/get-started/tutorials/create-an-app
#      https://docs.streamlit.io/develop/quick-reference/cheat-sheet 
#      https://docs.streamlit.io/develop/api-reference/charts/st.pyplot 

st.title("Power Generation of Philippine Fuel Sources")
st.write("Predicting Power Generated per Year (in GWh) Using Regression Models")

with st.spinner(text="Running regression models..."):
    results, X_test, Y_test, future_years, fuel_sources = RunModels()

# Options for regression model.
model_name = st.selectbox("Choose a model:", ["Linear", "Polynomial", "Ridge"])
selected_model = results[model_name]

# Options for fuel source.
fuel_options = [fuel for fuel in fuel_sources if fuel != "Grand Total"]  # Do not include "Grand Total"

fuel_name = st.selectbox("Choose a fuel source:", fuel_options)
fuel_index = fuel_sources.index(fuel_name)

# Metrics: ______________________________________________________________________________

# General regression metrics: _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

st.subheader(f"{model_name} Regression Metrics")

colA, colB, colC = st.columns(3)

# Print errors in columns.
with colA:
    st.metric("MAE", f"{selected_model['overall_metrics']['mae']:.3f}")

with colB:
    st.metric("MAPE", f"{selected_model['overall_metrics']['mape']:.3f}%")

with colC:
    st.metric("RMSE", f"{selected_model['overall_metrics']['rmse']:.3f}")

# Fuel source-specific metrics: _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

st.subheader(f"{fuel_name} Metrics for {model_name}")

fuel_metrics = selected_model["individual_metrics"][fuel_name]

colA, colB, colC = st.columns(3)

# Print errors in columns.
with colA:
    st.metric("MAE", f"{fuel_metrics['mae']:.3f}")

with colB:
    st.metric("MAPE", f"{fuel_metrics['mape']:.3f}%")

with colC:
    st.metric("RMSE", f"{fuel_metrics['rmse']:.3f}")

# Regression models comparison: _________________________________________________________

st.subheader("Evaluation Metric Comparison (per Model)")

# From main.py; plot Ccmparison uses matplotlib to graph the regression result.
# fig = PlotComparison(
#     results["Linear"]["overall_metrics"],
#     results["Polynomial"]["overall_metrics"],
#     results["Ridge"]["overall_metrics"]
# )

st.markdown(f"#### MAE")
st.bar_chart({"Linear": results["Linear"]["overall_metrics"]["mae"],
              "Polynomial": results["Polynomial"]["overall_metrics"]["mae"],
              "Ridge": results["Ridge"]["overall_metrics"]["mae"]})

st.markdown(f"#### MAPE")
st.bar_chart({"Linear": results["Linear"]["overall_metrics"]["mape"],
              "Polynomial": results["Polynomial"]["overall_metrics"]["mape"],
              "Ridge": results["Ridge"]["overall_metrics"]["mape"]})

st.markdown(f"#### RMSE")
st.bar_chart({"Linear": results["Linear"]["overall_metrics"]["rmse"],
              "Polynomial": results["Polynomial"]["overall_metrics"]["rmse"],
              "Ridge": results["Ridge"]["overall_metrics"]["rmse"]})

# Show figure.
# st.pyplot(fig)

# Fuel source-specific plot: ____________________________________________________________

years = X_test.flatten()                            # Flatten test years onto array.

actual = Y_test[:, fuel_index]                      # Actual values for selected fuel source.
predicted = selected_model["preds"][:, fuel_index]  # Predicted ...

st.subheader(f"Actual vs. Predicted {fuel_name} Power Generation")

# Combine values into 2 columns.
data = np.column_stack((actual, predicted))

st.line_chart(data, x_label = "Years", y_label = "Power Generation (Gwh)")