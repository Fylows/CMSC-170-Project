import matplotlib
matplotlib.use("Agg")   # Figures show in actual page.

import streamlit as st
from main import Linear, Polynomial, Ridge, LoadData

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

results, X_test, Y_test, future_years, fuel_sources = RunModels()