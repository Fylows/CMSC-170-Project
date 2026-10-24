import numpy as np

def parse_csv(file_path: str) -> np.array:
    with open(file_path, encoding='utf-16') as f:
        lines = f.readlines()
    
    
    data = np.genfromtxt([
        line.replace(',', '')
        for line in lines[1:]
    ])
    
    
    # returns data as a np array
    # rows as a list
    # columns as a list
    return data, data[1:,0], lines[0].strip().split(',')

def Mean_average_error(Y_true: dict, Y_pred: dict) -> float:
    """
    formula from: https://en.wikipedia.org/wiki/Mean_absolute_error
    Calculate the Mean Average Error (MAE) between true and predicted values.

    Parameters:
    Y_true (dict): A dictionary containing the true values.
    Y_pred (dict): A dictionary containing the predicted values.

    Returns:
    float: The Mean Average Error.
    """
    if not Y_true or not Y_pred:
        raise ValueError("Input dictionaries must not be empty.")

    if set(Y_true.keys()) != set(Y_pred.keys()):
        raise ValueError("Input dictionaries must have the same keys.")

    total_error = 0.0
    n = sum(len(v) for v in Y_true.values())
  
    for k, v in Y_true.items():
        if k not in Y_pred:
            raise ValueError(f"Key '{k}' is missing in the predicted values.")
        total_error += sum(abs(y_true - y_pred) for y_true, y_pred in zip(v, Y_pred[k]))

    mae = total_error / n
    return mae

def Mean_absolute_percentage_error(Y_true: dict, Y_pred: dict) -> float:
    """
    formula from: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    Calculate the Mean Absolute Percentage Error (MAPE) between true and predicted values.

    Parameters:
    Y_true (dict): A dictionary containing the true values.
    Y_pred (dict): A dictionary containing the predicted values.

    Returns:
    float: The Mean Absolute Percentage Error.
    """
    if not Y_true or not Y_pred:
        raise ValueError("Input dictionaries must not be empty.")

    if set(Y_true.keys()) != set(Y_pred.keys()):
        raise ValueError("Input dictionaries must have the same keys.")

    total_percentage_error = 0.0
    n = sum(len(v) for v in Y_true.values())

    for k, v in Y_true.items():
        if k not in Y_pred:
            raise ValueError(f"Key '{k}' is missing in the predicted values.")
        total_percentage_error += sum(abs((y_true - y_pred) / y_true) for y_true, y_pred in zip(v, Y_pred[k]) if y_true != 0)

    mape = (total_percentage_error / n) * 100
    return mape