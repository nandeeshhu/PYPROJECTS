import time
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class NiftyAnalyser:

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

    def plot_nifty50_prices(self):
        """
        Plots Nifty 50 stock prices, including Open, High, Low, Close, and Adjusted Close prices.

        Parameters:
        - df (pd.DataFrame): A Pandas DataFrame containing the stock price data.
                            It should have a 'Date' column as the index, and columns
                            for 'Open', 'High', 'Low', 'Close', and 'Adj Close' prices.

        Returns:
        None

        The function performs data preprocessing, including converting the 'Date' column to
        datetime format and setting it as the DataFrame index. It then creates subplots for
        each price category, plots the respective prices, and displays legends. The final plot
        includes Open, High, Low, Close, and Adjusted Close prices. The time taken for the
        entire operation is printed.

        Example:
        ```
        import pandas as pd
        import matplotlib.pyplot as plt
        import time

        # Assuming df is your DataFrame containing Nifty 50 stock prices
        plot_nifty50_prices(df)
        ```
        """
        # Data Preprocessing
        start_time = time.time()
        df = self.df.copy()
        df['Date'] = pd.to_datetime(df['Date'])

        df.set_index('Date', inplace=True)

        # Create subplots for each price category
        fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

        # Plot Open Price
        axes[0].plot(df.index, df['Open'], label='Open Price')
        axes[0].set_ylabel('Open Price')
        axes[0].set_title('Nifty 50 Stock Prices')

        # Plot High Price
        axes[1].plot(df.index, df['High'], label='High Price')
        axes[1].set_ylabel('High Price')

        # Plot Low Price
        axes[2].plot(df.index, df['Low'], label='Low Price')
        axes[2].set_ylabel('Low Price')

        # Plot Close Price
        axes[3].plot(df.index, df['Close'], label='Close Price')
        axes[3].set_ylabel('Close Price')

        # Plot Adjusted Close Price
        axes[4].plot(df.index, df['Adj Close'], label='Adj Close Price')
        axes[4].set_xlabel('Date')
        axes[4].set_ylabel('Adj Close Price')

        # Display legend for all subplots
        for ax in axes:
            ax.legend()

        plt.tight_layout()
        print("Time taken :", time.time() - start_time, "ms")
        plt.show()

    def plot_nifty50_prices_in_range(self, start_date, end_date):
        """
        Plots Nifty 50 stock prices, including Open, High, Low, Close, and Adjusted Close prices.

        Parameters:
        - df (pd.DataFrame): A Pandas DataFrame containing the stock price data.
                            It should have a 'Date' column as the index, and columns
                            for 'Open', 'High', 'Low', 'Close', and 'Adj Close' prices.

        Returns:
        None

        The function performs data preprocessing, including converting the 'Date' column to
        datetime format and setting it as the DataFrame index. It then creates subplots for
        each price category, plots the respective prices, and displays legends. The final plot
        includes Open, High, Low, Close, and Adjusted Close prices. The time taken for the
        entire operation is printed.

        Example:
        ```
        import pandas as pd
        import matplotlib.pyplot as plt
        import time

        # Assuming df is your DataFrame containing Nifty 50 stock prices
        plot_nifty50_prices(df)
        ```
        """
        start_time = time.time()
        df = self.df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        # Check if the provided date range is within the available data range
        min_date = df['Date'].min()
        max_date = df['Date'].max()

        if not (min_date <= pd.to_datetime(start_date) <= max_date and min_date <= pd.to_datetime(
                end_date) <= max_date):
            print(f"Data is only available from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}.")
            return
        df.set_index('Date', inplace=True)

        # Check if the specified date range is valid
        if start_date < df.index.min() or end_date > df.index.max():
            print("Error: Specified date range exceeds the available data.")
            return

        # Filter data for the specified date range
        df_subset = df[(df.index >= start_date) & (df.index <= end_date)]

        # Create subplots for each price category
        fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

        # Plot Open Price
        axes[0].plot(df_subset.index, df_subset['Open'], label='Open Price')
        axes[0].set_ylabel('Open Price')
        axes[0].set_title('Nifty 50 Stock Prices')

        # Plot High Price
        axes[1].plot(df_subset.index, df_subset['High'], label='High Price')
        axes[1].set_ylabel('High Price')

        # Plot Low Price
        axes[2].plot(df_subset.index, df_subset['Low'], label='Low Price')
        axes[2].set_ylabel('Low Price')

        # Plot Close Price
        axes[3].plot(df_subset.index, df_subset['Close'], label='Close Price')
        axes[3].set_ylabel('Close Price')

        # Plot Adjusted Close Price
        axes[4].plot(df_subset.index, df_subset['Adj Close'], label='Adj Close Price')
        axes[4].set_xlabel('Date')
        axes[4].set_ylabel('Adj Close Price')

        # Display legend for all subplots
        for ax in axes:
            ax.legend()

        plt.tight_layout()
        print("Time taken :", time.time() - start_time, "ms")
        plt.show()

    def plot_price_difference(self, start_date, end_date):
        """
        Plots the difference between High and Low prices over a specified date range.

        Parameters:
        - df (pd.DataFrame): A Pandas DataFrame containing the stock price data.
                            It should have a 'Date' column as the index, and columns
                            for 'Date', 'High', and 'Low' prices.
        - start_date (str or Timestamp): The start date of the desired date range.
        - end_date (str or Timestamp): The end date of the desired date range.

        Returns:
        None

        The function performs data preprocessing by selecting relevant columns ('Date', 'High', 'Low').
        It then converts the 'Date' column to datetime format and filters the data for the specified date range.
        The price difference (High - Low) is calculated and plotted over the selected period. The time taken for
        the entire operation is printed.

        Example:
        ```
        import pandas as pd
        import matplotlib.pyplot as plt
        import time

        # Assuming df is your DataFrame containing stock prices
        start_date = '2023-01-01'
        end_date = '2023-06-30'
        plot_price_difference(df, start_date, end_date)
        ```
        """
        start_time = time.time()
        # Data Preprocessing
        df = self.df.copy()
        # Convert the 'Date' column to a datetime format using .loc on a copy
        df.loc[:, 'Date'] = pd.to_datetime(df['Date'])
        # Check if the provided date range is within the available data range
        min_date = df['Date'].min()
        max_date = df['Date'].max()

        if not (min_date <= pd.to_datetime(start_date) <= max_date and min_date <= pd.to_datetime(
                end_date) <= max_date):
            print(f"Data is only available from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}.")
            return
        df = df[['Date', 'High', 'Low']]

        # Filter data for the specific time period
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        period_data = df[mask].copy()  # Create a copy explicitly

        # Calculate the price difference (High - Low) using .loc on the copy
        period_data.loc[:, 'Price Difference'] = period_data['High'] - period_data['Low']

        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(period_data['Date'], period_data['Price Difference'], label='Price Difference (High - Low)',
                 color='blue')

        plt.title(f'Price Difference (High - Low) from {start_date} to {end_date}')
        plt.xlabel('Date')
        plt.ylabel('Price Difference')
        plt.legend()
        plt.grid(True)
        print("Time taken :", time.time() - start_time, "ms")
        plt.show()

    def calculate_and_plot_volatility(self, window=20, plot_bollinger_bands=True):
        """
        Calculates and plots stock price volatility, optionally including Bollinger Bands.

        Parameters:
        - df (pd.DataFrame): A Pandas DataFrame containing stock price data.
                            It should have a 'Date' column and a 'Close' column.
        - window (int, optional): The size of the rolling window for calculating volatility.
                                  Defaults to 20.
        - plot_bollinger_bands (bool, optional): Whether to plot Bollinger Bands along with volatility.
                                                 Defaults to True.

        Returns:
        None

        The function calculates the rolling standard deviation (volatility) of the 'Close' prices
        using the specified window size. If 'plot_bollinger_bands' is True, it also calculates
        and plots the Bollinger Bands. The resulting plot displays the 'Close' prices, the rolling
        average (SMA), and optionally the Upper and Lower Bollinger Bands with shaded areas in
        between. If 'plot_bollinger_bands' is False, only the volatility is plotted over time.

        Example:
        ```
        import pandas as pd
        import matplotlib.pyplot as plt
        import time

        # Assuming df is your DataFrame containing stock prices
        calculate_and_plot_volatility(df, window=20, plot_bollinger_bands=True)
        ```
        """
        start_time = time.time()
        # Calculate rolling standard deviation (volatility)
        df = self.df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Volatility'] = df['Close'].rolling(window=window).std()

        if plot_bollinger_bands:
            # Calculate Bollinger Bands
            df['SMA'] = df['Close'].rolling(window=window).mean()
            df['Upper_Band'] = df['SMA'] + 2 * df['Volatility']
            df['Lower_Band'] = df['SMA'] - 2 * df['Volatility']

            # Plot Bollinger Bands
            plt.figure(figsize=(12, 6))
            plt.plot(df['Date'], df['Close'], label='Close Price')
            plt.plot(df['Date'], df['SMA'], label='SMA', linestyle='--')
            plt.plot(df['Date'], df['Upper_Band'], label='Upper Bollinger Band', linestyle='--')
            plt.plot(df['Date'], df['Lower_Band'], label='Lower Bollinger Band', linestyle='--')
            plt.fill_between(df['Date'], df['Upper_Band'], df['Lower_Band'], alpha=0.2)
            plt.title('Bollinger Bands for Price Volatility')
        else:
            # Plot only volatility
            plt.figure(figsize=(12, 6))
            plt.plot(df['Date'], df['Volatility'], label=f'Volatility (Rolling {window} Days)')
            plt.title(f'Price Volatility (Rolling {window} Days)')

        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True)
        print("Time taken :", time.time() - start_time, "ms")
        plt.show()

    def plot_trading_volume_with_spikes(self, start_date, end_date, moving_average_window,
                                        spikes_threshold=1.5):
        """
        Plots trading volume over time and identifies volume spikes using a moving average.

        Parameters:
        - df (pd.DataFrame): A Pandas DataFrame containing trading volume data.
                            It should have columns for dates ('col_date') and trading volume ('col_volume').
        - col_date (str): The name of the column containing dates.
        - col_volume (str): The name of the column containing trading volume data.
        - start_date (str or Timestamp): The start date for the plot and analysis.
        - end_date (str or Timestamp): The end date for the plot and analysis.
        - moving_average_window (int): The window size for the moving average calculation.
        - spikes_threshold (float, optional): The threshold for identifying volume spikes.
                                             Defaults to 1.5.

        Returns:
        None

        The function filters the DataFrame for the specified date range, replaces zero volume values with NaN,
        calculates the moving average of trading volume, and identifies volume spikes based on the specified
        threshold. It then plots the trading volume, the moving average, and a bar plot indicating volume spikes.
        The time taken for the entire operation is printed.

        Example:
        ```
        import pandas as pd
        import matplotlib.pyplot as plt
        import time

        # Assuming df is your DataFrame containing trading volume data
        col_date = 'Date'
        col_volume = 'Volume'
        start_date = '2023-01-01'
        end_date = '2023-06-30'
        moving_average_window = 10
        spikes_threshold = 1.5

        plot_trading_volume_with_spikes(df, col_date, col_volume, start_date, end_date,
                                         moving_average_window, spikes_threshold)
        ```
        """
        start_time = time.time()
        df = self.df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        # Check if the provided date range is within the available data range
        min_date = df['Date'].min()
        max_date = df['Date'].max()

        if not (min_date <= pd.to_datetime(start_date) <= max_date and min_date <= pd.to_datetime(
                end_date) <= max_date):
            print(f"Data is only available from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}.")
            return

        # Check if the start date is earlier than the available volume data
        min_volume_data_date = pd.to_datetime('2013-01-21')
        if start_date < min_volume_data_date:
            print("Volume data is not available before 2013-01-21. Data is available from that date onwards.")
            return
        # Filter the DataFrame for the specified date range
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

        # Replace zero volume values with NaN using .loc
        df.loc[df['Volume'] == 0, 'Volume'] = np.nan

        # Calculate moving average of trading volume while ignoring NaN values
        df['Volume_MA'] = df['Volume'].rolling(window=moving_average_window, min_periods=1).mean()

        # Identify and visualize spikes in trading activity using .loc
        df['Volume_Spike'] = 0
        df.loc[df['Volume'] > spikes_threshold * df['Volume_MA'], 'Volume_Spike'] = 1

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(df['Date'], df['Volume'], label='Trading Volume', color='blue')
        plt.plot(df['Date'], df['Volume_MA'], label=f'{moving_average_window}-Day MA', color='red')
        plt.title('Trading Volume Over Time')
        plt.xlabel('Date')
        plt.ylabel('Trading Volume')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.bar(df['Date'], df['Volume_Spike'], color='red', label=f'{spikes_threshold}-spike threshold')
        plt.title('Volume Spikes')
        plt.xlabel('Date')
        plt.ylabel('Volume Spike')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        print("Time taken :", time.time() - start_time, "ms")
        plt.show()

    def calculate_and_visualize_trading_volume(self, start_date, end_date,
                                               moving_average_window=20):
        """
        Calculates and visualizes trading volume and its correlation with other features.

        Parameters:
        - df (pd.DataFrame): A Pandas DataFrame containing trading volume data and additional features.
                            It should have columns for dates ('col_date'), trading volume ('col_volume'),
                            and other features.
        - col_date (str): The name of the column containing dates.
        - col_volume (str): The name of the column containing trading volume data.
        - start_date (str or Timestamp): The start date for the analysis.
        - end_date (str or Timestamp): The end date for the analysis.
        - moving_average_window (int, optional): The window size for the moving average calculation.
                                                 Defaults to 20.

        Returns:
        None

        The function checks if the start date is earlier than the available data, filters the DataFrame
        for the specified date range, replaces zero volume values with NaN, calculates the moving average
        of trading volume, and creates a correlation heatmap for the selected features. The time taken
        for the entire operation is printed.

        Example:
        ```
        import pandas as pd
        from visualize_utils import plot_correlation_heatmap  # Assuming you have a utility function
        import matplotlib.pyplot as plt
        import time

        # Assuming df is your DataFrame containing trading volume data
        col_date = 'Date'
        col_volume = 'Volume'
        start_date = '2023-01-01'
        end_date = '2023-06-30'
        moving_average_window = 20

        calculate_and_visualize_trading_volume(df, col_date, col_volume, start_date, end_date,
                                                moving_average_window)
        ```
        """
        start_time = time.time()
        df = self.df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        # Check if the start date is earlier than the available data
        # Check if the provided date range is within the available data range
        min_date = df['Date'].min()
        max_date = df['Date'].max()

        if not (min_date <= pd.to_datetime(start_date) <= max_date and min_date <= pd.to_datetime(
                end_date) <= max_date):
            print(f"Data is only available from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}.")
            return

        # Filter the DataFrame for the specified date range
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        # Replace zero volume values with NaN
        df['Volume'] = df['Volume'].replace(0, np.nan)

        # Calculate moving average of trading volume while ignoring NaN values
        df['Volume_MA'] = df['Volume'].rolling(window=moving_average_window, min_periods=1).mean()

        # Create a correlation heatmap
        self.plot_correlation_heatmap()
        plt.tight_layout()
        print("Time taken :", time.time() - start_time, "ms")
        plt.show()

    def plot_correlation_heatmap(self):
        """
        Plots a heatmap to visualize the correlation matrix of features in a DataFrame.

        Parameters:
        - df (pd.DataFrame): A Pandas DataFrame containing numerical features.

        Returns:
        None

        The function creates a heatmap to visualize the correlation matrix of features in the DataFrame.
        The correlation values are annotated in each cell of the heatmap. The time taken for the entire
        operation is not explicitly printed in this function but is included for consistency with the
        structure of the previous docstrings.

        Example:
        ```
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import time

        # Assuming df is your DataFrame containing numerical features
        plot_correlation_heatmap(df)
        plt.show()
        ```
        """
        start_time = time.time()
        df = self.df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        plt.figure(figsize=(10, 6))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')

    def plot_year_month_comparison(self, year):
        """
        Plots the difference between yearly and monthly average stock prices for a specified year.

        Parameters:
        - year (int): The year for which the comparison is to be made.

        Returns:
        None

        The function calculates the daily average stock price ('Avg') for each day, yearly average ('yearly_avg'),
        and monthly averages ('monthly_avg') for the given year. It then calculates the difference between yearly
        and monthly averages and plots the results using a bar chart. Each month is represented by a different
        color. The time taken for the entire operation is printed.

        If data is not available until the last date of the given year, it plots the difference only for the
        available months.

        Example:
        ```
        import pandas as pd
        import matplotlib.pyplot as plt
        import time

        # Assuming df is your DataFrame containing stock price data
        year = 2023
        plot_year_month_comparison(df, year)
        plt.show()
        """

        start_time = time.time()
        df = self.df.copy()
        # Ensure 'Date' column is in datetime format
        if not pd.api.types.is_datetime64_ns_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        # Extract the available year range
        min_year = df['Date'].dt.year.min()
        max_year = df['Date'].dt.year.max()

        # Check if the provided year is within the available data range
        if not (min_year <= year <= max_year):
            print(f"Data is only available for the years {min_year} to {max_year}.")
            return

        # Extract year from the 'Date' column
        df['Year'] = df['Date'].dt.year

        # Filter data for the given year
        df_year = df[df['Year'] == year]

        # Check if data is available until the last date of the given year
        if df_year['Date'].max() < pd.Timestamp(year, 12, 31):
            print(f"Data is not available until the last date of {year}. Plotting only for available months.")

        # Calculate daily average
        # df_year['Avg'] = df_year[['Open', 'High', 'Low', 'Close', 'Adj Close']].mean(axis=1)
        # df_year.loc[:, 'Avg'] = df_year[['Open', 'High', 'Low', 'Close', 'Adj Close']].mean(axis=1)
        df_year = df_year.assign(Avg=df_year[['Open', 'High', 'Low', 'Close', 'Adj Close']].mean(axis=1))

        # Calculate yearly average
        yearly_avg = df_year.groupby(['Year']).mean()

        # Calculate monthly averages for the given year
        monthly_avg = df_year.groupby(df_year['Date'].dt.month).mean()

        # Calculate the difference between yearly and monthly averages
        diff = yearly_avg['Avg'].values[0] - monthly_avg['Avg'].values

        # Define colors for each bar
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'brown', 'pink', 'gray', 'cyan',
                  'magenta']

        # Plotting
        plt.figure(figsize=(12, 8))

        month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                       "November", "December"]

        # Plot the difference for each month with different colors
        plt.bar(month_names[:len(diff)], diff, color=colors[:len(diff)], alpha=0.7,
                label=f'Diff Yearly - Monthly ({year})', align='center', width=0.4)

        plt.title(f'Yearly - Monthly Averages Difference ({year})')
        plt.xlabel('Month')
        plt.ylabel('Difference')
        plt.legend()
        print("Time taken:", time.time() - start_time, "ms")
        plt.show()

    def find_max_profit(self, start_date, end_date):
        """
        Finds the maximum profit one can make by purchasing after or on the start date
        and selling before or on the end date.

        Parameters:
        - df (pd.DataFrame): A Pandas DataFrame containing stock price data.
                            It should have columns for dates ('Date') and high prices ('High').

        Returns:
        float: The maximum profit that can be made within the specified period.

        The function prompts the user to input the start and end dates and then calculates
        the maximum profit by finding the minimum low price within the specified date range
        and the maximum high price after that date.

        Example:
        ```
        import pandas as pd

        # Assuming df is your DataFrame containing stock price data
        max_profit = find_max_profit_user_input(df)
        print(f'Maximum Profit: {max_profit}')
        ```
        """
        start_time = time.time()
        df = self.df.copy()
        # Ensure 'Date' column is in datetime format
        if not pd.api.types.is_datetime64_ns_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])

        # Check if the provided date range is within the available data range
        min_date = df['Date'].min()
        max_date = df['Date'].max()

        if not (min_date <= pd.to_datetime(start_date) <= max_date and min_date <= pd.to_datetime(
                end_date) <= max_date):
            print(f"Data is only available from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}.")
            return

        # Filter data for the specified date range
        df_subset = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        # Find the minimum low price within the specified date range
        min_low_price = df_subset['Low'].min()

        # Find the maximum high price after the date of the minimum low price
        max_high_price = df[df['Date'] > df[df['Low'] == min_low_price]['Date'].iloc[0]]['High'].max()

        # Calculate the maximum profit
        max_profit = max_high_price - min_low_price

        print("Maximum profit investor can make by investing in the given specific time is ", max_profit)
        print("Time taken:", time.time() - start_time, "ms")
