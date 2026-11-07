from lib import utility
from lib import Regressions
import numpy as np

# Print Prediction Function: ________________________________________

def PrintPrediction(years, preds, sources):           
    for year, pred in zip(years, preds):              # Iterate through years and predictions:
        print(f"\n{year[0]}:")                        #    Print predicted usage for each fuel source.

        for i in range(len(sources)):
            print(f"{sources[i]:<20}{pred[i]:,.2f}")

# Linear Regression: ________________________________________________

def Linear(X_train, Y_train, X_test, Y_test, fuel_sources, future_years):
    mean = X_train.mean(axis = 0)                                           # Get mean and standard dev. of training data.
    std = X_train.std(axis = 0)                         
    std[std == 0] = 1.0                                                     # Prevent division by zero.

    X_train_scaled = (X_train - mean) / std                                 # Scale training and testing data for gradient descent convergence.
    X_test_scaled = (X_test - mean) / std

    future_years_scaled = (future_years - mean) / std                       # Scale future years.
        
    final = Regressions.LinearRegression()                                  # Create model -> train on scaled training data -> predict.
    final.fit(X_train_scaled, Y_train)                       
    
    preds = final.predict(X_test_scaled)    
    future_preds = final.predict(future_years_scaled)                 

    # Get avg. error.
    mae = utility.Mean_average_error(               
        {fuel_sources[i]: Y_test[:, i] for i in range(len(fuel_sources))},  
        {fuel_sources[i]: preds[:, i] for i in range(len(fuel_sources))}
    )

    # Get mean abs. pctg. error.
    mape = utility.Mean_absolute_percentage_error(
        {fuel_sources[i]: Y_test[:, i] for i in range(len(fuel_sources))},
        {fuel_sources[i]: preds[:, i] for i in range(len(fuel_sources))}
    )

    # Print predictions for test years.
    for pred in preds:
        print(pred)

    print(f"\nMean Average Error (MAE): {mae}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")

    # Initial Model: _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    # print("\nInitial: _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _")
    # PrintPrediction(future_years, future_preds, fuel_sources)

    # Final Model (retrain on all years): _ _ _ _ _ _ _ _ _ _ _ _ _ _

    X_all = np.concatenate((X_train, X_test))                      # Put data back together by concatenation.
    Y_all = np.concatenate((Y_train, Y_test))

    mean_all = X_all.mean(axis = 0)                                # Get mean and standard dev. of all data.
    std_all = X_all.std(axis = 0)  
    std_all[std_all == 0] = 1.0                                    # Prevent division by zero.

    X_all_scaled = (X_all - mean_all) / std_all                    # Scale data and future years again.
    future_years_scaled_all = (future_years - mean_all) / std_all

    final_all = Regressions.LinearRegression()                     # Create model -> train on scaled data -> predict.
    final_all.fit(X_all_scaled, Y_all)   

    future_preds_all = final_all.predict(future_years_scaled_all)

    print("\nNew: _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _")
    PrintPrediction(future_years, future_preds_all, fuel_sources)

# Polynomial Regression______________________________________________
def Polynomial(X_train, Y_train, X_test, Y_test, fuel_sources, future_years, degree):
    mean = X_train.mean(axis = 0)                                   # Get mean and standard dev. of training data.
    std = X_train.std(axis = 0)                         
    std[std == 0] = 1.0                                             # prevent zero division    

    X_train_scaled = (X_train - mean) / std                         
    X_test_scaled = (X_test - mean) / std 
    
    final = Regressions.PolynomialRegression(degree = degree)
    final.fit(X_train_scaled, Y_train)

    preds = final.predict(X_test_scaled)
    
    # Get avg. error.
    mae = utility.Mean_average_error(               
        {fuel_sources[i]: Y_test[:, i] for i in range(len(fuel_sources))},  
        {fuel_sources[i]: preds[:, i] for i in range(len(fuel_sources))}
    )

    # Get mean abs. pctg. error.
    mape = utility.Mean_absolute_percentage_error(
        {fuel_sources[i]: Y_test[:, i] for i in range(len(fuel_sources))},
        {fuel_sources[i]: preds[:, i] for i in range(len(fuel_sources))}
    )

    # Print predictions for test years.
    for pred in preds:
        print(pred)

    print(f"\nMean Average Error (MAE): {mae}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")

    # Initial Model: _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    # print("\nInitial: _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _")
    # PrintPrediction(future_years, future_preds, fuel_sources)

    # Final Model (retrain on all years): _ _ _ _ _ _ _ _ _ _ _ _ _ _

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

    print("\nNew: _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _")
    PrintPrediction(future_years, future_preds_all, fuel_sources)

# Ridge Regression: _________________________________________________

def Ridge(X_train, Y_train, X_test, Y_test, fuel_sources):
    
    # get mean and std of training data, to scale both training and testing data
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0 # prevent division by zero for constant features

    # scale the training and testing data
    # since ridge regression is sensitive to the scale of the input features
    X_train_scaled = (X_train - mean) / std
    X_test_scaled  = (X_test - mean) / std
        
    final = Regressions.RidgeRegression(alpha=0.001) # hardcoded alpha for now, can be changed later
    final.fit(X_train_scaled, Y_train)
    preds = final.predict(X_test_scaled)
    
    mae = utility.Mean_average_error(
        {fuel_sources[i]: Y_test[:, i] for i in range(len(fuel_sources))},
        {fuel_sources[i]: preds[:, i] for i in range(len(fuel_sources))}
    )
    mape = utility.Mean_absolute_percentage_error(
        {fuel_sources[i]: Y_test[:, i] for i in range(len(fuel_sources))},
        {fuel_sources[i]: preds[:, i] for i in range(len(fuel_sources))}
    )

    for pred in preds:
        print(pred)

    print(f"\nMean Average Error (MAE): {mae}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")

# main(): ___________________________________________________________

def main():
    data, years, fuel_sources = utility.parse_csv("csv/Power Generation by Fuel Source.csv")
    data = data[:, 1:]  # remove year column
    years = np.asarray(years, dtype=float)
    data = np.asarray(data, dtype=float)
    
    assert len(years) == len(data), (len(years), len(data))  # rows must line up

    split = int(len(years) * 0.8)

    X_train = years[:split].reshape(-1, 1)
    X_test  = years[split:].reshape(-1, 1)
    Y_train = data[:split]
    Y_test  = data[split:]
    future_years = np.array([year for year in range(2021, 2031)]).reshape(-1, 1)  # Create array of years to predict.

    print("Linear: ____________________________________________________\n")
    Linear(X_train, Y_train, X_test, Y_test, fuel_sources, future_years)

    print("Polynomial: ____________________________________________________\n")
    Polynomial(X_train, Y_train, X_test, Y_test, fuel_sources, future_years, degree=4)
    

    print("\nRidge: _____________________________________________________\n")
    Ridge(X_train, Y_train, X_test, Y_test, fuel_sources)

if __name__ == "__main__":
    main()