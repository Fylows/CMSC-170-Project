import streamlit as st

st.title("Discussion")

# Discussion: ___________________________________________________________________________

st.divider()

st.header("...")



# Conclusion: ___________________________________________________________________________

st.divider()

st.header("Conclusion")

col_findings, col_interpretation = st.columns(2)
col_limitations, col_application = st.columns(2)

# Findings: _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

with col_findings.container(border = True):
    st.subheader("Findings")
    st.markdown("""
                Comparing the three regression models altogether, **linear and ridge** resulted in **almost identical
                evaluation metrics**. Both **outperformed polynomial** in terms of their **overall MAE and RMSE 
                values**, with said metrics being around **4,024 GWh and 6,664 GWh respectively**. When it came down
                to the **individual fuel sources**, however, the best model **varied**: **linear regression** gives the 
                best results for **coal, solar, and wind** sources; **polynomial** for **biomass and hydro**; and **ridge**
                for **geothermal, natural gas, and oil-based**.
                """)

# Interpretation: _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

with col_interpretation.container(border = True):
    st.subheader("Interpretation")
    st.markdown("""
                As **α** is set to **0.001** in **ridge regression**, it produces **similar predictions** to those of 
                **linear** due to its **small penalty** on weights (only having a **0.062-GWh MAE and 0.109 GWh-RMSE
                difference** between the two). Ridge's fit barely differs from linear's since the **penalty** can
                **shrink only a single slope** for each fuel source due to only having the **year** as the **lone
                input feature**.
                """)

# Limitations:  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
with col_limitations.container(border = True):
    st.subheader("Limitations")
    st.markdown("""
                Since the three models were based around **seven (7) test years** per fuel source (from 2014 to 2020), 
                **results may be noisy**. **Hyperparameters** such as α, learning rate, and number of iterations are also **fixed**. 
                Additionally, as mentioned in the About section, the project only has the **years** in the .csv as **input**, 
                and **does not** take into account **other external factors**.
                """)


# Conclusion: _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _



# _______________________________________________________________________________________

paginator_flex = st.container(horizontal=True,
    horizontal_alignment="distribute",
    vertical_alignment="center",     
    gap="medium")

if paginator_flex.button("Results Page"):
    st.switch_page("site_pages/results.py")
if paginator_flex.button("Predictor Page"):
    st.switch_page("site_pages/predictor.py")