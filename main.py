from lib import utility
from lib import Regressions

import numpy as np
import matplotlib.pyplot as mplot

iterations=10000
learning_rate = 0.001
# Plot Comparison Graph: ___________________________________________________

def PlotComparison(linear, polynomial, ridge):

    # Regression models used
    models = ["Linear", "Polynomial", "Ridge"]

    # Get overall MAE values for each model
    mae = [
        linear["mae"],
        polynomial["mae"],
        ridge["mae"]
    ]

    # Get overall MAPE values for each model
    mape = [
        linear["mape"],
        polynomial["mape"],
        ridge["mape"]
    ]

    # Get overall RMSE values for each model
    rmse = [
        linear["rmse"],
        polynomial["rmse"],
        ridge["rmse"]
    ]

    # Create figure(fig) and graphs(axes): 1 figure, 3 graphs
    fig, axes = mplot.subplots(1, 3, figsize=(15, 5))

    # MAE Graph
    axes[0].bar(models, mae)                            # plot MAE values for each regression model
    axes[0].set_title("Mean Absolute Error (MAE)")      # graph title
    axes[0].set_ylabel("Error")                         # label for Y axis
    axes[0].grid(axis="y", alpha=0.3)                   # set grid

    # MAPE Graph
    axes[1].bar(models, mape)                           # plot MAPE values for each regression model
    axes[1].set_title("Mean Absolute Percentage Error (MAPE)")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].grid(axis="y", alpha=0.3)

    # RMSE Graph
    axes[2].bar(models, rmse)                           # plot RMSE values for each regression model
    axes[2].set_title("Root Mean Squared Error (RMSE)")
    axes[2].set_ylabel("Error")
    axes[2].grid(axis="y", alpha=0.3)

    # figure title
    fig.suptitle("Regression Models Performance Comparison")

    # for display
    mplot.tight_layout()
    return fig

# Get Overall Metrics Function: _________________________________________________________
def GetOverallMetrics(keep, fuel_sources, y, preds):
    overall_metrics = {}                                   # Dict containing per fuel source's indiv. metrics to return.

    # Get avg. error.
    mae = utility.Mean_absolute_error(               
        {fuel_sources[i]: y[:, i] for i in keep},  
        {fuel_sources[i]: preds[:, i] for i in keep}
    )

    overall_metrics["mae"] = mae

    # Get mean abs. pctg. error.
    mape = utility.Mean_absolute_percentage_error(
        {fuel_sources[i]: y[:, i] for i in keep},
        {fuel_sources[i]: preds[:, i] for i in keep}
    )

    overall_metrics["mape"] = mape

    # Get root mean sqd. error.
    rmse = utility.Root_mean_squared_error(
        {fuel_sources[i]: y[:, i] for i in keep},
        {fuel_sources[i]: preds[:, i] for i in keep}
    )

    overall_metrics["rmse"] = rmse

    return overall_metrics

# Get Individual Metrics Function: ______________________________________________________

def GetIndividualMetrics(keep, fuel_sources, Y_test, preds):

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
        
        individual_metrics[fuel_sources[i]] = {"mae": mae_i, "mape": mape_i, "rmse": rmse_i}

    return individual_metrics

# Linear Regression: ____________________________________________________________________

def Linear(X_train, Y_train, X_test, Y_test, fuel_sources):
    mean = X_train.mean(axis = 0)                                                 # Get mean and standard dev. of training data.
    std = X_train.std(axis = 0)                         
    std[std == 0] = 1.0                                                           # Prevent division by zero.

    X_train_scaled = (X_train - mean) / std                                       # Scale training and testing data for gradient descent convergence.
    X_test_scaled = (X_test - mean) / std
        
    final = Regressions.LinearRegression(iterations=iterations, learning_rate=learning_rate)                                        # Create model -> train on scaled training data -> predict.
    final.fit(X_train_scaled, Y_train)                       
    
    preds = final.predict(X_test_scaled)    
    preds = np.maximum(preds, 0)

    keep = [i for i, s in enumerate(fuel_sources) if s != "Grand Total"]

    individual_metrics = GetIndividualMetrics(keep, fuel_sources, Y_test, preds)  # Individual ...

    
    # TRAINING PREDICTIONS AND METRICS
    train_preds = final.predict(X_train_scaled)
    train_preds = np.maximum(train_preds, 0)

    train_metrics = GetOverallMetrics(
        keep, fuel_sources, Y_train, train_preds
    )

    # TESTING PREDICTIONS AND METRICS
    test_preds = final.predict(X_test_scaled)
    test_preds = np.maximum(test_preds, 0)

    test_metrics = GetOverallMetrics(
        keep, fuel_sources, Y_test, test_preds
    )
   
    
    individual_metrics = GetIndividualMetrics(
        keep, fuel_sources, Y_test, test_preds
    )


    train_preds = final.predict(X_train_scaled)
    train_preds = np.maximum(train_preds, 0)
    train_metrics = GetOverallMetrics(
        keep, fuel_sources, Y_train, train_preds
    )

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_preds": train_preds,
        "test_preds": test_preds,
        "individual_metrics": individual_metrics,
        "model": final,
        "mean": mean,
        "std": std
    }

# Polynomial Regression__________________________________________________________________

def Polynomial(X_train, Y_train, X_test, Y_test, fuel_sources, degree):
    mean = X_train.mean(axis = 0)                                                      # get mean and standard dev. of training data
    std = X_train.std(axis = 0)                         
    std[std == 0] = 1.0                                                                # prevent zero division

    X_train_scaled = (X_train - mean) / std                         
    X_test_scaled = (X_test - mean) / std 
    
    final = Regressions.PolynomialRegression(degree = degree,iterations=iterations, learning_rate=learning_rate)
    final.fit(X_train_scaled, Y_train)

    preds = final.predict(X_test_scaled)
    preds = np.maximum(preds, 0)


    keep = [i for i, s in enumerate(fuel_sources) if s != "Grand Total"]
    
    overall_metrics = GetOverallMetrics(keep, fuel_sources, Y_test, preds)             # overall MAE, MAPE, and RMSE
    individual_metrics = GetIndividualMetrics(keep, fuel_sources, Y_test, preds)       # individual ...

    
    # TRAINING PREDICTIONS AND METRICS
    train_preds = final.predict(X_train_scaled)
    train_preds = np.maximum(train_preds, 0)

    train_metrics = GetOverallMetrics(
        keep, fuel_sources, Y_train, train_preds
    )

    # TESTING PREDICTIONS AND METRICS
    test_preds = final.predict(X_test_scaled)
    test_preds = np.maximum(test_preds, 0)

    test_metrics = GetOverallMetrics(
        keep, fuel_sources, Y_test, test_preds
    )
   
    
    individual_metrics = GetIndividualMetrics(
        keep, fuel_sources, Y_test, test_preds
    )


    train_preds = final.predict(X_train_scaled)
    train_preds = np.maximum(train_preds, 0)
    train_metrics = GetOverallMetrics(
        keep, fuel_sources, Y_train, train_preds
    )

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_preds": train_preds,
        "test_preds": test_preds,
        "individual_metrics": individual_metrics,
        "model": final,
        "mean": mean,
        "std": std
    }

# Ridge Regression: _____________________________________________________________________

def Ridge(X_train, Y_train, X_test, Y_test, fuel_sources):
    
    # get mean and std of training data, to scale both training and testing data
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0 # prevent division by zero for constant features

    # scale the training and testing data
    # since ridge regression is sensitive to the scale of the input features
    X_train_scaled = (X_train - mean) / std
    X_test_scaled  = (X_test - mean) / std
        
    final = Regressions.RidgeRegression(alpha=0.001,iterations=iterations, learning_rate=learning_rate)                              # hardcoded alpha for now, can be changed later
    final.fit(X_train_scaled, Y_train)

    keep = [i for i, s in enumerate(fuel_sources) if s != "Grand Total"]

    # TRAINING PREDICTIONS AND METRICS
    train_preds = final.predict(X_train_scaled)
    train_preds = np.maximum(train_preds, 0)

    train_metrics = GetOverallMetrics(
        keep, fuel_sources, Y_train, train_preds
    )

    # TESTING PREDICTIONS AND METRICS
    test_preds = final.predict(X_test_scaled)
    test_preds = np.maximum(test_preds, 0)

    test_metrics = GetOverallMetrics(
        keep, fuel_sources, Y_test, test_preds
    )
   
    
    individual_metrics = GetIndividualMetrics(
        keep, fuel_sources, Y_test, test_preds
    )


    train_preds = final.predict(X_train_scaled)
    train_preds = np.maximum(train_preds, 0)
    train_metrics = GetOverallMetrics(
        keep, fuel_sources, Y_train, train_preds
    )

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_preds": train_preds,
        "test_preds": test_preds,
        "individual_metrics": individual_metrics,
        "model": final,
        "mean": mean,
        "std": std
    }

# LoadData(): ___________________________________________________________________________

def LoadData():
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

    return X_train, X_test, Y_train, Y_test, fuel_sources