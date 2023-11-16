import time
from datetime import timedelta, datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class NiftyPredictor:
    def __init__(self):
        # Initialize an empty DataFrame
        self.df = pd.DataFrame()

        # Nifty index symbol and date range for web scraping
        ticker = '^NSEI'
        start_date = int(time.mktime(datetime(2008, 1, 1, 23, 59).timetuple()))
        end_date = datetime.now() - timedelta(days=1)
        end_date = end_date.replace(hour=23, minute=59, second=0, microsecond=0)
        end_date = int(time.mktime(end_date.timetuple()))
        interval = '1d'  # 1d, 1m

        # Web scraping attempt
        query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_date}&period2={end_date}&interval={interval}&events=history&includeAdjustedClose=true'

        try:
            self.df = pd.read_csv(query_string)
            self.df.to_csv('nifty.csv')
        except pd.errors.EmptyDataError:
            print("Web data is empty. Using offline data available.")
            # If loading from the web fails, try loading from the CSV file
            self.df = pd.read_csv('nifty.csv')
        except Exception as e:
            print(f"Failed to load data from the web. Error: {e}.")
            print("Using offline data available.\n")
            # If loading from the web fails, try loading from the CSV file
            self.df = pd.read_csv('nifty.csv')

    def predict_stock_prices(self, target_date):
        """
        Predicts stock prices (High, Low, Close, Adj Close) at a specified target date
        using linear regression based on historical data.

        Parameters:
        - df (pd.DataFrame): A Pandas DataFrame containing historical stock price data.
                            It should have a column for dates ('col_date') and columns
                            for 'High', 'Low', 'Close', and 'Adj Close' prices.
        - col_date (str): The name of the column containing dates.
        - target_date (str or Timestamp): The target date for which to predict stock prices.

        Returns:
        dict: A dictionary containing predicted stock prices for 'High', 'Low', 'Close',
              and 'Adj Close' at the specified target date.

        The function ensures that the 'col_date' column is in datetime format and sorts
        the DataFrame by date. It then iterates through each price column, handles missing
        values by filling them with the mean of existing values, prepares the data for linear
        regression, and predicts the price at the target date. The predicted prices are returned
        in a dictionary.

        Example:
        ```
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        import time

        # Assuming df is your DataFrame containing historical stock prices
        col_date = 'Date'
        target_date = '2023-12-31'
        predictions = predict_stock_prices(df, col_date, target_date)
        print(predictions)
        ```
        """
        df = self.df.copy()
        start_time = time.time()
        # Ensure 'Date' column is in datetime format
        if not pd.api.types.is_datetime64_ns_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])

        # Sort the DataFrame by date
        df = df.sort_values(by='Date')

        # Extract historical dates and prices
        historical_dates = df['Date']
        cols_to_predict = ['High', 'Low', 'Close', 'Adj Close']

        predicted_prices = {}

        for col in cols_to_predict:
            # Check for and handle missing values in the current column
            if df[col].isna().any():
                # Fill missing values with the mean of the existing values
                mean_price = df[col].mean()
                df[col] = df[col].fillna(mean_price)

            # Prepare the data for linear regression
            X = [[(d - historical_dates.iloc[0]).days] for d in historical_dates]
            y = df[col].tolist()

            # Initialize the linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Predict the price at the target date
            days_since_start = (target_date - historical_dates.iloc[0]).days
            predicted_price = model.predict([[days_since_start]])

            predicted_prices[col] = predicted_price[0]
        print("Time taken :", time.time() - start_time, "ms")

        return predicted_prices

    def future_plot(self, start_date, end_date):
        """
        Predicts and plots future stock prices (High, Low, Close, Adj Close) over a specified date range.

        Parameters:
        - start_date (str or Timestamp): The start date for predicting and plotting future prices.
        - end_date (str or Timestamp): The end date for predicting and plotting future prices.

        Returns:
        None

        The function ensures that the 'Date' column is in datetime format. It then checks if the specified
        start_date is in the future by comparing it with the maximum date in the dataset. If the start_date
        is not in the future, the function prints a message and returns without plotting.

        It then iteratively predicts stock prices for each day within the specified date range using the
        'predict_stock_prices' function. The predicted prices are plotted over time for 'High', 'Low',
        'Close', and 'Adj Close'. The time taken for the entire operation is printed.

        Example:
        ```
        import pandas as pd
        from datetime import datetime, timedelta
        import matplotlib.pyplot as plt
        import time

        # Assuming df is your DataFrame containing historical stock prices
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)
        future_plot(start_date, end_date)
        ```
        """
        start_time = time.time()
        df = self.df.copy()
        max_date = pd.to_datetime(df['Date'].max())

        # Ensure 'Date' column is in datetime format
        if not pd.api.types.is_datetime64_ns_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])

        # Check if start_date is in the future
        if start_date <= max_date:
            print("The specified start date is not in the future. Please provide a start date after the last date in the dataset.")
            return

        current_date = start_date  # Initialize the current date with the start date
        cols_to_plot = ['High', 'Low', 'Close', 'Adj Close']
        predicted_prices = {col: [] for col in cols_to_plot}

        while current_date <= end_date:
            predicted_prices_current = self.predict_stock_prices_(current_date)

            for col in cols_to_plot:
                predicted_prices[col].append(predicted_prices_current[col])

            # Increment the current date by one day
            current_date += timedelta(days=1)

        # Plot the predicted prices
        fig, axs = plt.subplots(len(cols_to_plot), figsize=(10, 8))
        fig.suptitle('Predicted Prices Over Time')

        for i, col in enumerate(cols_to_plot):
            axs[i].plot(pd.date_range(start=start_date, periods=len(predicted_prices[col]), freq='D'),
                        predicted_prices[col], label=f'Predicted {col}')
            axs[i].set_ylabel(f'{col} Price')
            axs[i].legend()

        plt.xlabel('Date')
        print("Time taken:", time.time() - start_time, "ms")
        plt.show()

    def predict_stock_prices_(self, target_date):
        """
        Predicts stock prices (High, Low, Close, Adj Close) at a specified target date
        using linear regression based on historical data.

        Parameters:
        - df (pd.DataFrame): A Pandas DataFrame containing historical stock price data.
                            It should have a column for dates ('col_date') and columns
                            for 'High', 'Low', 'Close', and 'Adj Close' prices.
        - col_date (str): The name of the column containing dates.
        - target_date (str or Timestamp): The target date for which to predict stock prices.

        Returns:
        dict: A dictionary containing predicted stock prices for 'High', 'Low', 'Close',
              and 'Adj Close' at the specified target date.

        The function ensures that the 'col_date' column is in datetime format and sorts
        the DataFrame by date. It then iterates through each price column, handles missing
        values by filling them with the mean of existing values, prepares the data for linear
        regression, and predicts the price at the target date. The predicted prices are returned
        in a dictionary.

        Example:
        ```
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        import time

        # Assuming df is your DataFrame containing historical stock prices
        col_date = 'Date'
        target_date = '2023-12-31'
        predictions = predict_stock_prices(df, col_date, target_date)
        print(predictions)
        ```
        """
        df = self.df.copy()
        # Ensure 'Date' column is in datetime format
        if not pd.api.types.is_datetime64_ns_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])

        # Sort the DataFrame by date
        df = df.sort_values(by='Date')

        # Extract historical dates and prices
        historical_dates = df['Date']
        cols_to_predict = ['High', 'Low', 'Close', 'Adj Close']

        predicted_prices = {}

        for col in cols_to_predict:
            # Check for and handle missing values in the current column
            if df[col].isna().any():
                # Fill missing values with the mean of the existing values
                mean_price = df[col].mean()
                df[col] = df[col].fillna(mean_price)

            # Prepare the data for linear regression
            X = [[(d - historical_dates.iloc[0]).days] for d in historical_dates]
            y = df[col].tolist()

            # Initialize the linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Predict the price at the target date
            days_since_start = (target_date - historical_dates.iloc[0]).days
            predicted_price = model.predict([[days_since_start]])

            predicted_prices[col] = predicted_price[0]

        return predicted_prices

