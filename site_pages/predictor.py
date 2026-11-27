import streamlit as st
import numpy as np
from main import Linear, Polynomial, Ridge, LoadData
import altair as alt

def clear_text():
    st.session_state["start_year"] = ""
    st.session_state["end_year"] = ""
# RunModels(): __________________________________________________________________________

@st.cache_data
def LoadCached():
    return LoadData()

@st.cache_data
def RunModels():
    X_train, X_test, Y_train, Y_test, fuel_sources = LoadCached()

    linear_results = Linear(X_train, Y_train, X_test, Y_test, fuel_sources)
    polynomial_results = Polynomial(
        X_train, Y_train, X_test, Y_test, fuel_sources, degree=4
    )
    ridge_results = Ridge(X_train, Y_train, X_test, Y_test, fuel_sources)

    results = {
        "Linear": linear_results,
        "Polynomial": polynomial_results,
        "Ridge": ridge_results
    }

    return results, fuel_sources

st.title("Try it yourself!")
model = st.pills("Choose a model:", ["Linear", "Polynomial", "Ridge"], default="Linear")
if model == [] or model == None:
    st.info("Select a model")
    st.stop()

flex = st.container(horizontal=True)

start_txt = flex.text_input("Enter the starting year", placeholder="e.g. 2020", max_chars=4, key="start_year")
end_txt = flex.text_input("Enter the end year", placeholder="e.g. 2026", max_chars=4, key="end_year")
st.button("Clear", on_click=clear_text)

if not (start_txt and end_txt):
    st.info("Enter dates")
    st.stop()

if not (start_txt.isdigit() and end_txt.isdigit()):
    st.info("Years must be whole numbers, e.g. 2026")
    st.stop()

start_yr, end_yr = int(start_txt), int(end_txt)

if start_yr > end_yr:
    st.info("Enter a valid range")
    st.stop()

years = np.arange(start_yr, end_yr + 1).reshape(-1, 1)   # inclusive, shape (n, 1)
results, fuel_sources = RunModels()
selected_model = results[model]

years_scaled = (years - selected_model["mean"]) / selected_model["std"]

predictions = selected_model["model"].predict(years_scaled)
predictions = np.maximum(predictions, 0)

year_list = years.ravel()

chart_data = []
for j, source in enumerate(fuel_sources):
    if source == "Grand Total":          # skip it so it doesn't squash the other lines
        continue
    for i, year in enumerate(year_list):
        chart_data.append({
            "Years": int(year),
            "Fuel Source": source,
            "Power Generation": float(predictions[i, j]),
        })

# make two charts, one bar chart for single years, and another line chart for multi dates   
if len(year_list) == 1:
    chart = alt.Chart(alt.Data(values=chart_data)).mark_bar().encode(
        x=alt.X("Fuel Source:N", title="Fuel Source", sort="-y"),
        y=alt.Y("Power Generation:Q", title="Power Generation (GWh)", axis=alt.Axis(format=",d")),
        color=alt.Color("Fuel Source:N", legend=None),
        tooltip=[
            alt.Tooltip("Years:Q", title="Year", format="d"),
            alt.Tooltip("Fuel Source:N", title="Fuel Source"),
            alt.Tooltip("Power Generation:Q", title="Power Generation (GWh)", format=",.0f"),
        ],
    ).properties(height=450, title=f"{model} ({int(year_list[0])})")
else:
    chart = alt.Chart(alt.Data(values=chart_data)).mark_line(point=True).encode(
        x=alt.X("Years:Q", title="Years", axis=alt.Axis(format="d")),
        y=alt.Y("Power Generation:Q", title="Power Generation (GWh)", axis=alt.Axis(format=",d")),
        color=alt.Color("Fuel Source:N", title="Legend"),
        tooltip=[
            alt.Tooltip("Years:Q", title="Year", format="d"),
            alt.Tooltip("Fuel Source:N", title="Fuel Source", ),
            alt.Tooltip("Power Generation:Q", title="Power Generation (GWh)", format=",.0f"),
        ],
    ).properties(height=450, title=model)

st.altair_chart(chart, use_container_width=True, key=f"chart_{model}")


paginator_flex = st.container(horizontal=True,
    horizontal_alignment="distribute",
    vertical_alignment="center",     
    gap="medium")
if paginator_flex.button("Discussion Page"):
    st.switch_page("site_pages/discussion.py")
if paginator_flex.button("References Page"):
    st.switch_page("site_pages/references.py")