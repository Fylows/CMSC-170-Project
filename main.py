from lib import utility



if __name__ == "__main__":
    true = {
        "2024": [3.0, 5.0, 2.5],
    }
    
    predicted = {
        "2024": [2.5, 5.0, 3.0],
    }
    print(f"MAE: {round(utility.Mean_average_error(true, predicted) * 100, 3)}%")
    print(f"MAPE: {round(utility.Mean_absolute_percentage_error(true, predicted), 3)}%")