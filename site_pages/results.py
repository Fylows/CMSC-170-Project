import altair as alt
import numpy as np

import matplotlib
matplotlib.use("Agg")   # Figures show in actual page.

import matplotlib.pyplot as plt

import streamlit as st
from main import Linear, Polynomial, Ridge, LoadData, PlotComparison

# RunModels(): __________________________________________________________________________

@st.cache_data
def RunModels():
    X_train, X_test, Y_train, Y_test, fuel_sources = LoadData()

    linear_results = Linear(X_train, Y_train, X_test, Y_test, fuel_sources)
    polynomial_results = Polynomial(X_train, Y_train, X_test, Y_test, fuel_sources, degree=4)
    ridge_results = Ridge(X_train, Y_train, X_test, Y_test, fuel_sources)

    results = {"Linear": linear_results, "Polynomial": polynomial_results, "Ridge": ridge_results}
    return results, X_train, X_test, Y_train, Y_test, fuel_sources

# _______________________________________________________________________________________

# Streamlit
# Ref: https://docs.streamlit.io/get-started/tutorials/create-an-app
#      https://docs.streamlit.io/develop/quick-reference/cheat-sheet 
#      https://docs.streamlit.io/develop/api-reference/charts/st.pyplot 


st.set_page_config(layout = "wide")

st.title("Results")
st.write("This section shows the results of the models after they have been trained, and tested on the seven (7) years in our dataset.")

with st.spinner(text="Running regression models..."):
    results, X_train, X_test, Y_train, Y_test, fuel_sources = RunModels()


tab_training, tab_testing= st.tabs(["Training", "Testing"])
with tab_training:
    model_name = st.selectbox("Choose a model:", ["Linear", "Polynomial", "Ridge"], key="model_select_results_3")
    selected_model = results[model_name]
    # Overall metrics, pooled over all fuel sources except Grand Total.
    cols = [i for i, f in enumerate(fuel_sources) if f != "Grand Total"]
    
    st.subheader(f"{model_name} Regression Training Metrics")
    colA, colB, colC = st.columns(3)
    colA.metric("MAE", f"{selected_model['train_metrics']['mae']:.3f}")
    colB.metric("RMSE", f"{selected_model['train_metrics']['rmse']:.3f}")
    colC.metric("MAPE", f"{selected_model['train_metrics']['mape']:.3f}%")

    st.subheader("Evaluation Metric Comparison (per Model)")
    flex = st.container(horizontal=True)
    train_overall = {
        m: results[m]["train_metrics"] for m in ["Linear", "Polynomial", "Ridge"]
    }
    flex.bar_chart({m: {"MAE": v["mae"], "RMSE": v["rmse"]} for m, v in train_overall.items()}, stack=False)
    flex.bar_chart({m: {"MAPE": v["mape"]} for m, v in train_overall.items()}, stack=False)

    st.header(f"{model_name} Regression Model per Fuel Source Metrics")
    fuel_options = [f for f in fuel_sources if f != "Grand Total"]
    fuel_name = st.pills("Choose a fuel source:", fuel_options, default=fuel_options[0], key="model_select_fuel_2")

    if fuel_name is None:
        st.info("Please select a fuel source.")
    else:
        fuel_index = fuel_sources.index(fuel_name)
        train_years = X_train.flatten().astype(int)
        actual = Y_train[:, fuel_index]
        predicted = selected_model['train_preds'][:, fuel_index]

        fm = selected_model['individual_metrics'][fuel_name]
        st.subheader(f"{fuel_name} Metrics")
        colA, colB, colC = st.columns(3)
        colA.metric("MAE", f"{fm['mae']:.3f}")
        colB.metric("RMSE", f"{fm['rmse']:.3f}")
        colC.metric("MAPE", f"{fm['mape']:.3f}%")

        # Fuel source-specific plot: ____________________________________________________________

        st.subheader(f"Actual vs. Predicted {fuel_name} Power Generation")
        chart_data = []
        for i in range(len(train_years)):
            chart_data.append({"Years": int(train_years[i]), "Type": "Actual", "Power Generation": float(actual[i])})
            chart_data.append({"Years": int(train_years[i]), "Type": "Predicted", "Power Generation": float(predicted[i])})
    
        # create line graph
        chart = alt.Chart(alt.Data(values=chart_data)).mark_line(point=True).encode(                   # mark_line for line graph
            x=alt.X("Years:Q", title="Years", axis=alt.Axis(format="d")), 
            y=alt.Y("Power Generation:Q", title="Power Generation (GWh)", axis=alt.Axis(format="d")),
            color=alt.Color("Type:N", title="Legend"),
    
            # info displayed when hovered
            tooltip=[   
                alt.Tooltip("Years:Q", title="Year", format="d"),
                alt.Tooltip("Type:N", title="Type"),
                alt.Tooltip("Power Generation:Q", title="Power Generation", format="d")
            ]
        ).properties(
            height=450,  # set graph height to 450
            title=fuel_name
        )
    
        # show chart
        st.altair_chart(chart, use_container_width=True, key=f"chart_{fuel_name}_2")
    
    
with tab_testing:
    # Options for regression model.
    model_name = st.selectbox("Choose a model:", ["Linear", "Polynomial", "Ridge"])
    selected_model = results[model_name]

    # General regression metrics: ____________________________________________________________

    st.subheader(f"{model_name} Regression Testing Metrics")
    colA, colB, colC = st.columns(3)

    # Print errors in columns.
    with colA:
        st.metric("MAE", f"{selected_model['test_metrics']['mae']:.3f}")

    with colB:
        st.metric("RMSE", f"{selected_model['test_metrics']['rmse']:.3f}")

    with colC:
        st.metric("MAPE", f"{selected_model['test_metrics']['mape']:.3f}%")

    # Regression models comparison: _________________________________________________________

    st.subheader("Evaluation Metric Comparison (per Model)")
    flex = st.container(horizontal = True)
    # outer keys = bars within each group (models), inner keys = x-axis (metrics)
    data1 = {
        m: {
            "MAE":  results[m]['test_metrics']["mae"],
            "RMSE": results[m]['test_metrics']["rmse"],
        }
        for m in ["Linear", "Polynomial", "Ridge"]
    }

    data2 = {
        m: {
            "MAPE": results[m]['test_metrics']["mape"],
        }
        for m in ["Linear", "Polynomial", "Ridge"]
    }

    flex.bar_chart(data1, stack=False)
    flex.bar_chart(data2, stack=False)

    st.header(f"{model_name} Regression Model per Fuel source Metrics")

    # Options for fuel source.
    fuel_options = [fuel for fuel in fuel_sources if fuel != "Grand Total"]  # Do not include "Grand Total"
    fuel_name = st.pills("Choose a fuel source:", fuel_options, default=fuel_options[0])
    if fuel_name is None:
        st.info("Please select a fuel source.")
        st.stop()
    fuel_index = fuel_sources.index(fuel_name)

    # Fuel source-specific metrics: __________________________________________________________

    st.subheader(f"{fuel_name} Metrics")

    fuel_metrics = selected_model["individual_metrics"][fuel_name]

    colA, colB, colC = st.columns(3)

    # Print errors in columns.
    with colA:
        st.metric("MAE", f"{fuel_metrics['mae']:.3f}")

    with colB:
        st.metric("RMSE", f"{fuel_metrics['rmse']:.3f}")

    with colC:
        st.metric("MAPE", f"{fuel_metrics['mape']:.3f}%")

    # Fuel source-specific plot: ____________________________________________________________

    years = X_test.flatten().astype(int)               # make years int
    actual = Y_test[:,fuel_index]                      # get actual and pred vals for fuel src 
    predicted = selected_model["test_preds"][:,fuel_index]
    st.subheader(f"Actual vs. Predicted {fuel_name} Power Generation")

    chart_data = []

    # append actual and pred vals as dict to data list
    for i in range(len(years)):
        chart_data.append({"Years":int(years[i]), "Type":"Actual", "Power Generation":float(actual[i])})
        chart_data.append({"Years":int(years[i]), "Type":"Predicted", "Power Generation":float(predicted[i])})

    # create line graph
    chart = alt.Chart(alt.Data(values=chart_data)).mark_line(point=True).encode(                   # mark_line for line graph
        x=alt.X("Years:Q", title="Years", axis=alt.Axis(format="d")), 
        y=alt.Y("Power Generation:Q", title="Power Generation (GWh)", axis=alt.Axis(format="d")),
        color=alt.Color("Type:N", title="Legend"),

        # info displayed when hovered
        tooltip=[   
            alt.Tooltip("Years:Q", title="Year", format="d"),
            alt.Tooltip("Type:N", title="Type"),
            alt.Tooltip("Power Generation:Q", title="Power Generation", format="d")
        ]
    ).properties(
        height=450,  # set graph height to 450
        title=fuel_name
    )

    # show chart
    st.altair_chart(chart, use_container_width = True, key=f"chart_{fuel_name}")

paginator_flex = st.container(horizontal=True,
    horizontal_alignment="distribute",
    vertical_alignment="center",     
    gap="medium")

if paginator_flex.button("Methodology Page"):
    st.switch_page("site_pages/methodology.py")
if paginator_flex.button("Discussion Page"):
    st.switch_page("site_pages/discussion.py")


