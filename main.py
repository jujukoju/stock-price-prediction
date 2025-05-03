import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR


def get_data(file_name):
    try:
        data = pd.read_csv(file_name)
        if 'Date' not in data.columns or 'Adj Close' not in data.columns:
            raise ValueError("CSV file must contain 'Date' and 'Adj Close' columns")

        data['Date'] = pd.to_datetime(data['Date'])
        start_date = data['Date'].min()
        dates = (data['Date'] - start_date).dt.days.values
        prices = data['Adj Close'].values

        return dates, prices, start_date
    except FileNotFoundError:
        print(f"Error! File {file_name} not found")
        return np.array([]), np.array([]), None


def predict_price(dates, prices, x):
    if len(dates) == 0 or len(prices) == 0:
        print("Error: No data available")

    dates = np.reshape(dates, (len(dates), 1))

    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='green', label='RBF Model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    x = np.array([[x]])
    return svr_rbf.predict(x)[0]


def shares_calculation(predicted_price, investment):
    if predicted_price <= 0:
        return 0
    return investment/predicted_price


def main():
    dates, prices, start_date = get_data("apple_stock.csv")

    if len(dates) == 0 or start_date is None:
        print("Can't proceed because...data loading issues")
        return

    date_input = input("Pick a prediction date (format: YYYY-MM-DD)")
    prediction_date = pd.to_datetime(date_input)
    x = (prediction_date-start_date).days

    investment = float(input("Enter the amount to invest ($): "))
    if investment <= 0:
        raise ValueError("Investment amount must be positive")

    rbf_price = predict_price(dates, prices, x)

    if rbf_price is not None:
        rbf_shares = shares_calculation(rbf_price, investment)

        print(f"Predicted Price (RBF): ${rbf_price:.2f}, Shares: {rbf_shares:.2f}")
    else:
        print("Prediction failed due to invalid data")


if __name__ == "__main__":
    main()
