from lib import utility



if __name__ == "__main__":
    true = {
        "2024": [3.0, 5.0, 2.5],
    }
    
    predicted = {
        "2024": [2.5, 5.0, 3.0],
    }
    
    data, rows, columns = utility.parse_csv("/home/yel/Documents/projects/cmsc-170/csv/Power Generation by Fuel Source.csv")
    rows = (int(row) for row in rows.tolist()) # convert rows to integers
    columns = list(columns[0].split())[1:] # exclude the 'Year' Column
    
    print(f"CSV Data: {data}")
    print(f"Rows: {list(rows)}")
    print(f"Columns: {columns}")
    
    # print(f"MAE: {round(utility.Mean_average_error(true, predicted) * 100, 3)}%")
    # print(f"MAPE: {round(utility.Mean_absolute_percentage_error(true, predicted), 3)}%")