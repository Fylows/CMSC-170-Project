import streamlit as st

st.set_page_config(
    layout="wide"
)

st.title("Power Generation of Philippine Fuel Sources")
st.markdown("## Predicting Power Generated per Year (in GWh) Using Regression Models")
st.markdown("Made by **Francis Miguel Rabulan**, **Ljiel Saplan**, and **Deigna Phoebe Marie Serrano**")

# About Section: ________________________________________________________________________

st.divider()

st.header("About the Project")

col_problem, col_motivation = st.columns(2)

# Problem Statement:  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

with col_problem.container(border = True):
    st.subheader("Problem Statement")
    st.markdown("""
                The Department of Energy (DOE) has collected [data](https://data.gov.ph/index/public/dataset/Power%20Generation%20by%20Fuel%20Source,%201990-2020/mhb14dc0-8tio-af0u-6nac-euxrn6bz48rm) 
                on the **yearly power generation of various fuel sources** in the Philippines from **1990 to 2020**, measured in gigawatt 
                hours (GWh). Given this, how well can **different regression models** predict **future generation** for each source?

                This project:
                - Implements **linear**, **polynomial**, and **ridge** regression;
                - Trains them on earlier years and tests them on recent ones;
                - Compares them using evaluation metrics: **MAE**, **MAPE**, and **RMSE** (overall and per fuel source); and
                - Forecasts generation for **2021 to 2030**.

                The main objective of this project is to determine which of the three models **predicts best overall**, and whether the **best 
                changes** depending on the fuel source.
                """)

# Motivation: _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

with col_motivation.container(border = True):
    st.subheader("Motivation")
    st.markdown("""
                **Energy planning efforts** in the country may be supported by getting an idea of **how much** each fuel source can generate in 
                **future years**. Since a source's trend **may differ** from another's, one model may **not be apt** for all of them.

                Thus, this project compares three regression models with **varying flexibility** to see where each one works well and fails, with
                varying factors considered alongside doing these (such as gradient descent and regularization).

                It is important to note that these **only extend past trends** based on DOE's data, and **do not** account for existing policies, prices, or 
                power plants; this project is **solely a comparison** of linear, polynomial, and ridge regression.
                """)

# Dataset Description: __________________________________________________________________

st.divider()

st.header("Dataset Description")

col_description, col_preparation = st.columns(2)

# Dataset Information Statement:  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

with col_description.container(border = True):
    st.subheader("Information")
    st.markdown("""
                - **Source**: [Department of Energy - "Power Generation by Fuel Source, 1990-2020"](https://data.gov.ph/index/public/dataset/Power%20Generation%20by%20Fuel%20Source,%201990-2020/mhb14dc0-8tio-af0u-6nac-euxrn6bz48rm)
                - **Size**: Observations for **eight (8) fuel sources** (biomass, coal, geothermal, hydro, natural gas, oil-based, solar, and wind) across 
                **thirty-one (31) years** (from 1990 to 2020).
                - **Features**: The **year**, with **polynomial regression** expanding it up to **degree 4** (x⁴).
                - **Target**: The **power generated** per year, in **gigawatt-hours** (GWh), per fuel source.
                """)

# Dataset Preprocessing:  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

with col_preparation.container(border = True):
    st.subheader("Preprocessing Steps")
    st.markdown("""
                1. **Parsing the `.csv`**: The file is read as **UTF-16** and commas (serving as thousands separators) are removed.
                2. **Train-Test Split**: The **first twenty-four (24) years** (1990 to 2013) serve as **training years**, while the **latter seven (7)** 
                (2014 to 2020) are used for **testing**.
                3. **Scaling**: To keep gradient descent stable, the years are **standardized** in `main.py` (subtract the mean and divide by the standard
                deviation) using the training ones. The same values are applied to the testing and future years.
                4. **Non-negative Predictions**: Since power generation **cannot** be negative, negative predictions are **set to zero (0)**.
                5. **Final Forecast**: After evaluation, the three models are **retrained** to forecast the **next decade** after the available data (2021 
                to 2030).
                """)

# _______________________________________________________________________________________

paginator_flex = st.container(horizontal=True,
    horizontal_alignment="distribute",
    vertical_alignment="center",     
    gap="medium")

if paginator_flex.button("References Page"):
    st.switch_page("site_pages/references.py")
if paginator_flex.button("Methodology Page"):
    st.switch_page("site_pages/methodology.py")