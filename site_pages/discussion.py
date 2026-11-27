import streamlit as st

st.title("Discussion")
st.write("This section elaborates upon the results of the regression models' run.")

# Discussion: ___________________________________________________________________________

st.divider()

st.header("Performance Analysis")
col_linear, col_polynomial = st.columns(2)

with col_linear.container(border = True):
    st.subheader("Linear Regression")
    st.markdown("""
                Linear regression generally performed the best out of all the three models, achieving a test MAE of approximately 4,024 and a test RMSE of 6,665.
                However, its main limitation is that **it assumes a linear relationship between the year and electricity generation**. 
                
                This makes it difficult to capture sudden increases, decreases, or changing trends in certain fuel sources.
                For example, natural gas generation increased dramatically around 2001-2002. A straight-line model struggles to represent this transition accurately.
                """)
    
with col_polynomial.container(border = True):
    st.subheader("Polynomial Regression")
    st.markdown("""
                Polynomial regression achieved the lowest training MAE and RMSE among the three models. However, its testing performance was considerably worse.
                
                This suggests that the model may be overfitting the training data. Because we used a fourth-degree polynomial, the model can capture more complex patterns, but it can also become too closely fitted to historical fluctuations.
                """)
    
with st.container(border = True):
    st.subheader("Ridge Regression")
    st.markdown("""
    Ridge regression produced almost identical results to linear regression, with a small alpha value of 0.001.
    But upon adjusting it to 1, our training errors increased slightly, but our testing MAE improved from approximately 4,024 to 3,895, while testing MAPE decreased from 57.98% to 55.58%.

    This suggests that stronger regularization helped the model generalize slightly better according to these metrics. However, testing RMSE increased from approximately 6,665 to 6,896, indicating that some larger prediction errors may still be present.
    """)

# Conclusion: ___________________________________________________________________________

st.divider()

st.header("Conclusion")
st.write("This section reiterates and established what was found and could be improved.")

col_findings, col_interpretation = st.columns(2)

with col_findings.container(border = True):
    st.subheader("Findings")
    st.markdown("""
                Comparing the three regression models altogether, linear and ridge resulted in **close
                evaluation metrics**. Both outperformed polynomial in terms of their **overall MAE and RMSE 
                values**, with said metrics having a difference of only around **25 GWh** and **26 GWh** respectively. 
                When it came down to the individual fuel sources, however, the **best model varied** (in terms of MAE): 
                linear regression gives the best results for coal, solar, and wind sources; polynomial for biomass and 
                hydro; and ridge for geothermal, natural gas, and oil-based.
                """)

with col_interpretation.container(border = True):
    st.subheader("MAPE Limitations")
    st.markdown("""
                Our training MAPE is **unusually high**, particularly for linear and ridge regression.
                As we discovered in the natural gas diagnostics, actual generation values were **extremely small** during some
                early years. Even relatively moderate prediction errors can result in **enormous percentage errors** when the 
                actual value is close to zero.
                
                Therefore, MAPE alone **does not** accurately represent the overall performance of our models. MAE and RMSE 
                provide additional context.
                """)

with st.container(border = True):
    st.subheader("Conclusion")
    st.markdown("""
                Since the three models were based around seven (7) test years per fuel source (from 2014 to 2020), 
                **results may be noisy**. Hyperparameters such as α, learning rate, and number of iterations are also **fixed**. 
                Additionally, as mentioned in the About section, the project only has the **years** in the .csv as **input**, 
                and **does not** take into account other external factors. Further implementations may have the aforementioned 
                hyperparameters (as well as other ones like the degree in polynomial regression) **tuned** as opposed to being fixed.
                """)
    
# _______________________________________________________________________________________

paginator_flex = st.container(horizontal=True,
    horizontal_alignment="distribute",
    vertical_alignment="center",     
    gap="medium")

if paginator_flex.button("Results Page"):
    st.switch_page("site_pages/results.py")
if paginator_flex.button("Predictor Page"):
    st.switch_page("site_pages/predictor.py")