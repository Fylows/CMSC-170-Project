from lib import utility
from lib import Regressions
import numpy as np

# Get Overall Metrics Function: _________________________________________________________

def GetOverallMetrics(keep, fuel_sources, Y_test, preds):
    # Get avg. error.
    mae = utility.Mean_absolute_error(               
        {fuel_sources[i]: Y_test[:, i] for i in keep},  
        {fuel_sources[i]: preds[:, i] for i in keep}
    )

    # Get mean abs. pctg. error.
    mape = utility.Mean_absolute_percentage_error(
        {fuel_sources[i]: Y_test[:, i] for i in keep},
        {fuel_sources[i]: preds[:, i] for i in keep}
    )

    # Get root mean sqd. error.
    rmse = utility.Root_mean_squared_error(
        {fuel_sources[i]: Y_test[:, i] for i in keep},
        {fuel_sources[i]: preds[:, i] for i in keep}
    )

    print("\n[OVERALL EVAL. METRICS]")
    print(f"\n{'Mean Absolute Error (MAE):':<40} {mae:,.2f}")
    print(f"{'Mean Absolute Percentage Error (MAPE):':<40} {mape:,.2f}")
    print(f"{'Root Mean Squared Error (RMSE):':<40} {rmse:,.2f}")

    return mae, mape, rmse

# Get Individual Metrics Function: ______________________________________________________

def GetIndividualMetrics(keep, fuel_sources, Y_test, preds):
    print("\n[INDIVIDUAL EVAL. METRICS]\n")
    print(f"{'Fuel Source':<20}{'MAE':>12}{'MAPE':>12}{'RMSE':>12}")

    individual_metrics = {}                                           # Dict containing per fuel source's indiv. metrics to return.

    for i in keep:
        mae_i = utility.Mean_absolute_error(
            {fuel_sources[i]: Y_test[:, i]},
            {fuel_sources[i]: preds[:, i]}
        )

        mape_i = utility.Mean_absolute_percentage_error(
            {fuel_sources[i]: Y_test[:, i]},
            {fuel_sources[i]: preds[:, i]}
        )

        rmse_i = utility.Root_mean_squared_error(
            {fuel_sources[i]: Y_test[:, i]},
            {fuel_sources[i]: preds[:, i]}
        )

        print(f"{fuel_sources[i]:<20}{mae_i:>12,.2f}{mape_i:>12,.2f}{rmse_i:>12,.2f}")
        
        individual_metrics[fuel_sources[i]] = {"mae": mae_i, "mape": mape_i, "rmse": rmse_i}

    return individual_metrics

# Print Prediction Function: ____________________________________________________________

def PrintPrediction(years, preds, sources):
    print("\n[YEARLY PREDICTIONS]")

    keep = [i for i, s in enumerate(sources) if s != "Grand Total"]  # Remove grand total since that isn't accurate.
    
    for year, pred in zip(years, preds):                             # Iterate through all years and their corresponding prediction.
        print(f"\n{year[0]}:")
        print(f"{'Source':<20}{'Generated Power (GWh)':}")

        for i in keep:
            print(f"{sources[i]:<20}{pred[i]:,.2f}")

        print("- - - - - - - - - - - - - - - -")
        print(f"{'Grand Total':<20}{pred[keep].sum():,.2f}")         # Sum of all the predictions.

# Linear Regression: ____________________________________________________________________

def Linear(X_train, Y_train, X_test, Y_test, fuel_sources, future_years):
    mean = X_train.mean(axis = 0)                                                 # Get mean and standard dev. of training data.
    std = X_train.std(axis = 0)                         
    std[std == 0] = 1.0                                                           # Prevent division by zero.

    X_train_scaled = (X_train - mean) / std                                       # Scale training and testing data for gradient descent convergence.
    X_test_scaled = (X_test - mean) / std
        
    final = Regressions.LinearRegression()                                        # Create model -> train on scaled training data -> predict.
    final.fit(X_train_scaled, Y_train)                       
    
    preds = final.predict(X_test_scaled)    
    preds = np.maximum(preds, 0)

    keep = [i for i, s in enumerate(fuel_sources) if s != "Grand Total"]

    overall_metrics = GetOverallMetrics(keep, fuel_sources, Y_test, preds)        # Overall MAE, MAPE, and RMSE.
    individual_metrics = GetIndividualMetrics(keep, fuel_sources, Y_test, preds)  # Individual ...

    # Final Model (retrain on all years): _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    X_all = np.concatenate((X_train, X_test))                              # Put data back together by concatenation.
    Y_all = np.concatenate((Y_train, Y_test))

    mean_all = X_all.mean(axis = 0)                                        # Get mean and standard dev. of all data.
    std_all = X_all.std(axis = 0)  
    std_all[std_all == 0] = 1.0                                            # Prevent division by zero.

    X_all_scaled = (X_all - mean_all) / std_all                            # Scale data and future years again.
    future_years_scaled_all = (future_years - mean_all) / std_all

    final_all = Regressions.LinearRegression()                             # Create model -> train on scaled data -> predict.
    final_all.fit(X_all_scaled, Y_all)   

    future_preds_all = final_all.predict(future_years_scaled_all)
    future_preds_all = np.maximum(future_preds_all, 0)
    
    PrintPrediction(future_years, future_preds_all, fuel_sources)

    return {"overall_metrics": overall_metrics, 
            "preds": preds,
            "future_preds_all": future_preds_all,
            "individual_metrics": individual_metrics}

# Polynomial Regression__________________________________________________________________

def Polynomial(X_train, Y_train, X_test, Y_test, fuel_sources, future_years, degree):
    mean = X_train.mean(axis = 0)                                                      # get mean and standard dev. of training data
    std = X_train.std(axis = 0)                         
    std[std == 0] = 1.0                                                                # prevent zero division

    X_train_scaled = (X_train - mean) / std                         
    X_test_scaled = (X_test - mean) / std 
    
    final = Regressions.PolynomialRegression(degree = degree)
    final.fit(X_train_scaled, Y_train)

    preds = final.predict(X_test_scaled)
    preds = np.maximum(preds, 0)

    keep = [i for i, s in enumerate(fuel_sources) if s != "Grand Total"]
    
    overall_metrics = GetOverallMetrics(keep, fuel_sources, Y_test, preds)             # overall MAE, MAPE, and RMSE
    individual_metrics = GetIndividualMetrics(keep, fuel_sources, Y_test, preds)       # individual ...

    # Final Model (retrain on all years): _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    X_all = np.concatenate((X_train, X_test))
    Y_all = np.concatenate((Y_train, Y_test))

    mean_all = X_all.mean(axis = 0)                                
    std_all = X_all.std(axis = 0)  
    std_all[std_all == 0] = 1.0                                    

    X_all_scaled = (X_all - mean_all) / std_all                    
    future_years_scaled_all = (future_years - mean_all) / std_all

    final_all = Regressions.PolynomialRegression(degree=degree)                     
    final_all.fit(X_all_scaled, Y_all)   

    future_preds_all = final_all.predict(future_years_scaled_all)
    future_preds_all = np.maximum(future_preds_all, 0)

    PrintPrediction(future_years, future_preds_all, fuel_sources)

    return {"overall_metrics": overall_metrics, 
            "preds": preds,
            "future_preds_all": future_preds_all,
            "individual_metrics": individual_metrics}

# Ridge Regression: _____________________________________________________________________

def Ridge(X_train, Y_train, X_test, Y_test, fuel_sources, future_years):
    
    # get mean and std of training data, to scale both training and testing data
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0 # prevent division by zero for constant features

    # scale the training and testing data
    # since ridge regression is sensitive to the scale of the input features
    X_train_scaled = (X_train - mean) / std
    X_test_scaled  = (X_test - mean) / std
        
    final = Regressions.RidgeRegression(alpha=0.001)                              # hardcoded alpha for now, can be changed later
    final.fit(X_train_scaled, Y_train)

    preds = final.predict(X_test_scaled)
    preds = np.maximum(preds, 0)

    keep = [i for i, s in enumerate(fuel_sources) if s != "Grand Total"]
    
    overall_metrics = GetOverallMetrics(keep, fuel_sources, Y_test, preds)        # Overall MAE, MAPE, and RMSE.
    individual_metrics = GetIndividualMetrics(keep, fuel_sources, Y_test, preds)  # Individual ...

    # Final Model (retrain on all years): _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    X_all = np.concatenate((X_train, X_test))
    Y_all = np.concatenate((Y_train, Y_test))

    mean_all = X_all.mean(axis = 0)                                
    std_all = X_all.std(axis = 0)  
    std_all[std_all == 0] = 1.0                                    

    X_all_scaled = (X_all - mean_all) / std_all                    
    future_years_scaled_all = (future_years - mean_all) / std_all

    final_all = Regressions.RidgeRegression(alpha=0.001)                     
    final_all.fit(X_all_scaled, Y_all)   

    future_preds_all = final_all.predict(future_years_scaled_all)
    future_preds_all = np.maximum(future_preds_all, 0)

    PrintPrediction(future_years, future_preds_all, fuel_sources)

    return {"overall_metrics": overall_metrics, 
            "preds": preds,
            "future_preds_all": future_preds_all,
            "individual_metrics": individual_metrics}

# main(): _______________________________________________________________________________

def main():
    data, years, fuel_sources = utility.parse_csv("csv/Power Generation by Fuel Source.csv")
    data = data[:, 1:]                                                            # remove year column
    years = np.asarray(years, dtype=float)
    data = np.asarray(data, dtype=float)
    
    assert len(years) == len(data), (len(years), len(data))                       # rows must line up

    split = int(len(years) * 0.8)

    X_train = years[:split].reshape(-1, 1)
    X_test  = years[split:].reshape(-1, 1)
    Y_train = data[:split]
    Y_test  = data[split:]
    future_years = np.array([year for year in range(2021, 2031)]).reshape(-1, 1)  # Create array of years to predict.

    print("Linear: _________________________________________________________")
    linear_results = Linear(X_train, Y_train, X_test, Y_test, fuel_sources, future_years)

    print("\nPolynomial: _____________________________________________________")
    polynomial_results = Polynomial(X_train, Y_train, X_test, Y_test, fuel_sources, future_years, degree=4)

    print("\nRidge: __________________________________________________________")
    ridge_results = Ridge(X_train, Y_train, X_test, Y_test, fuel_sources, future_years)

if __name__ == "__main__":
    main()