from lib import utility
from lib import Regressions
import numpy as np

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
    print(f"Mean Average Error (MAE): {mae}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")

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

    Ridge(X_train, Y_train, X_test, Y_test, fuel_sources)

if __name__ == "__main__":
    main()

    

