from niftyanalyser import NiftyAnalyser
from niftypridictor import NiftyPredictor
import pandas as pd


def main():
    nfa = NiftyAnalyser()
    nfp = NiftyPredictor()
    while True:
        print("Enter key for the following information")
        print("1: To get plot of variation of nifty price from 2008 to 2023.")
        print("2: To get plot of specific period of time")
        print("3: To get expected price of given upcoming date")
        print("4: To get plot of difference of lowest and highest price for specific time period")
        print("5: To get estimated plot of upcoming days")
        print("6: To Calculate and plot price volatility")
        print("7: Plot trading volume with spikes for the specified period")
        print("8: To get plot of Correlation Heatmap.")
        print("9: To get year wise average year - average month plot")
        print("10: To get maximum profit possible.")
        print("'end': To end the program")

        key = input()
        if key == '1':
            nfa.plot_nifty50_prices()
            print("\n\n")
            continue;

        elif key == '2':
            start_date = pd.to_datetime(input("Enter start date in yyyy-mm-dd format"))
            end_date = pd.to_datetime(input("Enter start date in yyyy-mm-dd format"))
            nfa.plot_nifty50_prices_in_range(start_date, end_date)
            print("\n\n")
            continue

        elif key == '3':
            target_date = pd.to_datetime(input("Enter upcoming date in yyyy-mm-dd format to get estimated price"))

            # Call the function to predict and print all prices at the target date
            predicted_prices = nfp.predict_stock_prices(target_date)
            for col, price in predicted_prices.items():
                print(f"Predicted {col} price at {target_date}: {price:.2f}")
            print("\n\n")

        elif key == '4':
            start_date = pd.to_datetime(input("Enter start date in yyyy-mm-dd format"))
            end_date = pd.to_datetime(input("Enter end date in yyyy-mm-dd format"))
            nfa.plot_price_difference(start_date, end_date)
            print("\n\n")

        elif key == '5':
            start_date = pd.to_datetime(input("Enter start date in yyyy-mm-dd format"))
            end_date = pd.to_datetime(input("Enter end date in yyyy-mm-dd format"))
            nfp.future_plot(start_date, end_date)
            print("\n\n")

        elif key == '6':
            nfa.calculate_and_plot_volatility(window=20)
            print("\n\n")

        elif key == '7':
            start_date = pd.to_datetime(input("Enter start date in yyyy-mm-dd format"))
            end_date = pd.to_datetime(input("Enter end date in yyyy-mm-dd format"))
            moving_average_window = int(input("Enter the moving average window in days."))
            # Plot trading volume with spikes for the specified period
            nfa.plot_trading_volume_with_spikes(start_date, end_date,
                                                moving_average_window,
                                                spikes_threshold=1.5)
            print("\n\n")

        elif key == '8':
            start_date = pd.to_datetime(input("Enter start date in yyyy-mm-dd format"))
            end_date = pd.to_datetime(input("Enter end date in yyyy-mm-dd format"))
            nfa.calculate_and_visualize_trading_volume(start_date, end_date)
            print("\n\n")

        elif key == '9':
            year = int(input("Enter the year"))
            nfa.plot_year_month_comparison(year)
            print("\n\n")

        elif key == '10':
            start_date = pd.to_datetime(input("Enter start date in yyyy-mm-dd format: "))
            end_date = pd.to_datetime(input("Enter end date in yyyy-mm-dd format: "))
            nfa.find_max_profit(start_date, end_date)
            print("\n\n")

        elif key == 'end':
            print("Exiting program...")
            break

        else:
            print("Enter valid key")


if __name__ == '__main__':
    main()
