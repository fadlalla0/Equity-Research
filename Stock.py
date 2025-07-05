from datetime import datetime
import pandas as pd
import requests
import os
import json
import helper_functions as hf
from pathlib import Path
pd.set_option("display.float_format", "{:,.2f}".format)

class Stock:
    def __init__(self, symbol: str, api_key: str):
        self.symbol = symbol
        self.api_key = api_key
        self.profile = None
        self.sector = None
        self.industry = None
        self.quote = None
        self.price = None
        self.change = None
        self.eps = None

    def load_data(self):
        self.profile = self.get_profile()
        if self.profile:
            self.sector = self.profile.get('sector')
            self.industry = self.profile.get('industry')
        self.quote = self.get_quote()
        if self.quote:
            self.price = self.quote.get('price')
            self.change = self.quote.get('change')
        self.eps = self.get_eps_dilluted()

    def make_request(self, endpoint: str):
        """
        Helper method to make API requests
        """
        base_url = "https://financialmodelingprep.com/stable"
        
        separator = '&' if '?' in endpoint else '?'
        
        response = requests.get(f"{base_url}/{endpoint}{separator}apikey={self.api_key}&symbol={self.symbol}")
        if response.status_code == 200:
            return response.json()
        return None
    
    def get_profile(self):
        """
        Get the profile for the stock
        """
        data = self.make_request("profile?")
        for sample in data:
            if sample['symbol'] == self.symbol:
                return sample
        return None

    def get_quote(self):
        """
        Get the quote for the stock
        """
        return self.make_request('quote?')[0]

    def get_statement(self, statement: str, period: str = "annual", growth: bool = False, reported_currency: bool = False):
        """
        Get a statement from the balance sheet, income statement, or cash flow statement
        statement_type: Type of financial statement ('income-statement', 'balance-sheet-statement', 'cash-flow-statement')
        """
        if growth:
            statement = pd.DataFrame(self.make_request(f"{statement}-growth?period={period}&limit=1000&"))
        else:
            statement = pd.DataFrame(self.make_request(f"{statement}?period={period}&limit=1000&"))
        statement = statement.set_index(['fiscalYear', 'period', 'filingDate'])
        if reported_currency:
            statement = statement.drop(['date', 'symbol', 'cik', 'acceptedDate'], axis=1)
        else: 
            statement = statement.drop(['date', 'symbol', 'reportedCurrency', 'cik', 'acceptedDate'], axis=1)

        return statement
    
    def get_ratios(self, ttm: bool = True):
        """
        Get the ratios for the stock
        """
        if ttm:
            return pd.DataFrame(self.make_request(f"ratios-ttm?"))
        else:
            return pd.DataFrame(self.make_request(f"ratios?limit=100&")).drop(['symbol', 'date', 'reportedCurrency'], axis=1).set_index(['fiscalYear', 'period'])
        
    def get_key_metrics(self, ttm: bool = True):
        """
        Get the key metrics for the stock
        """
        if ttm:
            return pd.DataFrame(self.make_request(f"key-metrics-ttm?"))
        else:
            return pd.DataFrame(self.make_request(f"key-metrics?limit=100&")).drop(['symbol', 'date', 'reportedCurrency'], axis=1).set_index(['fiscalYear', 'period'])
        
    def get_revenue_segmentation(self, product: bool = True):
        """
        Get the revenue segmentation for the stock
        """
        if product:
            data = self.make_request(f"revenue-product-segmentation?")
            revenue_segmentation = {"date": [], "data": []}
        else:
            data = self.make_request(f"revenue-geographic-segmentation?")
            revenue_segmentation = {"date": [], "data": []}
        for point in data:
            revenue_segmentation["date"].append(point['date'])
            revenue_segmentation["data"].append(point['data'])
        revenue_segmentation = pd.DataFrame(revenue_segmentation)
        segmentation = dict(revenue_segmentation['data'])
        segmentation = pd.DataFrame(segmentation).transpose().set_axis(revenue_segmentation['date'], axis=0)
        segmentation = segmentation.reset_index(drop=False)
        segmentation = segmentation.melt(id_vars='date', var_name='segment', value_name='revenue')
        # Remove '.' and everything after it in the 'revenue' column (as string), then convert to float
        segmentation['revenue'] = segmentation['revenue'].fillna(0).astype(str).str.replace(r'\..*', '', regex=True)
        segmentation['revenue'] = segmentation['revenue'].replace('', '0').astype(float)

        return segmentation

    def get_insider_trading(self):
        """
        Get the insider trading for the stock
        """
        return pd.DataFrame(self.make_request(f"insider-trading/search?"))
    
    def get_analyst_estimates(self, period: str = "annual"):
        """
        Get the analyst estimates for the stock
        """
        return pd.DataFrame(self.make_request(f"analyst-estimates?period={period}&"))
    
    def get_historical_price(self, from_date: str = "1970-01-01", to_date: str = datetime.now().strftime("%Y-%m-%d"), update=False):
        """
        Get the historical price for the stock. Data is cached in a 'prices' folder.
        If cached data exists, only fetches new data from the last available date.
        If the latest date in cached data is the last Friday, no new requests are made.
        """

        # Create prices directory if it doesn't exist
        prices_dir = Path("prices")
        prices_dir.mkdir(exist_ok=True)
        
        # Define file path for this symbol
        file_path = prices_dir / f"{self.symbol}.csv"
        
        # Check if cached data exists
        if file_path.exists() and not update:
            # Read existing data
            cached_data = pd.read_csv(file_path, index_col='date', parse_dates=True)
            latest_date = cached_data.index.max()
            last_modification = datetime.fromtimestamp(file_path.stat().st_mtime)
            time_since_modification = datetime.now() - last_modification
            x = 6

            if time_since_modification.total_seconds() > (60 * 60 * x):
                # Get new data from latest date to to_date
                new_data = pd.DataFrame(self.make_request(
                    f"historical-price-eod/full?from={latest_date.strftime('%Y-%m-%d')}&to={to_date}&"))
                new_data['date'] = pd.to_datetime(new_data['date'])
                new_data = new_data.set_index('date')
                # Remove the latest date from cached data to avoid duplicates
                cached_data = cached_data[cached_data.index < latest_date]
                
                # Combine and sort data
                data = pd.concat([cached_data, new_data])
                data = data.sort_index()
                
                # Save updated data
                data.to_csv(file_path)
                data = data[(data.index >= from_date) & (data.index <= to_date)]
                return data
            cached_data = cached_data[(cached_data.index >= from_date) & (cached_data.index <= to_date)]
            return cached_data
        
        # If no cached data exists, get all data
        data = pd.DataFrame(self.make_request(f"historical-price-eod/full?&from=1970-01-01&to={datetime.now().strftime('%Y-%m-%d')}&"))
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')

        
        # Save data
        data.to_csv(file_path)
        data = data[(data.index >= from_date) & (data.index <= to_date)]
        data = data.sort_index()
        return data
    
    def get_eps_dilluted(self):
        """
        Calculate the diluted earnings per share (EPS) for the last 4 quarters.
        
        Returns:
            float: The sum of diluted EPS for the last 4 quarters
        """
        eps = self.get_statement('income-statement', 'quarterly')
        eps = eps[['weightedAverageShsOutDil', 'netIncome']]
        eps['epsDiluted'] = eps['netIncome'] / eps['weightedAverageShsOutDil']
        eps = eps['epsDiluted'][:4].sum()
        return eps

    def get_financial_scores(self):
        """
        Get the financial scores for the stock
        """
        data = pd.DataFrame(self.make_request(f"financial-scores?"))
        if data.empty:
            return pd.DataFrame()
        data = data.drop('reportedCurrency', axis=1)
        data = data.set_index('symbol')
        return data

    def calculate_ttm(self, 
                 statement_type: str,
                 column_name: str,
                 average: bool = False) -> pd.DataFrame:
        """
        Calculate rolling 4-quarter sum or average for a specific column in financial statements.
        
        Args:
            statement_type (str): Type of financial statement ('income-statement', 'balance-sheet-statement', 'cash-flow-statement')
            column_name (str): Name of the column to calculate rolling sum/average for
            average (bool, optional): If True, calculate rolling average. If False, calculate rolling sum. Defaults to False.
            
        Returns:
            pd.DataFrame: DataFrame containing the original data and rolling 4-quarter sum/average
            
        Raises:
            ValueError: If no data is available for the symbol or if the column is not found
        """
        # Get quarterly data
        df = self.get_statement(statement_type, period='quarter')
        
        if df is None or df.empty:
            raise ValueError(f"No data available for {self.symbol}")
            
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the financial statement")
            
        # Sort by index (date) in descending order (most recent first)
        df = df.sort_index(ascending=True)
        
        # Calculate rolling 4-quarter sum
        if average:
            df[f'{column_name}_4q_rolling'] = df[column_name].rolling(window=4, min_periods=1).mean()
        else:
            df[f'{column_name}_4q_rolling'] = df[column_name].rolling(window=4, min_periods=1).sum()
        
        # Select relevant columns and sort by date
        result = df[[column_name, f'{column_name}_4q_rolling']].sort_index(ascending=False)
        result = result.reset_index().drop(['fiscalYear', 'period'], axis=1).set_index(['filingDate'])
        
        return result
    
    def calculate_margins(self, period: str = 'annual'):
        """
        Calculate and return a DataFrame with various profit margins:
        Gross Margin, Operating Margin, EBITDA Margin, Pre-Tax Margin, Net Margin.
        Returns:
            pd.DataFrame: DataFrame with margin columns indexed by date (filingDate)
        """
        # Get the income statement (TTM if available, else annual)
        income = self.get_statement('income-statement', period=period)
        if income is None or income.empty:
            return pd.DataFrame()
        # Ensure all required columns exist
        required_cols = [
            'revenue',
            'grossProfit',
            'operatingIncome',
            'ebitda',
            'incomeBeforeTax',
            'netIncome'
        ]
        for col in required_cols:
            if col not in income.columns:
                raise ValueError(f"Column '{col}' not found in the income statement")
        # Calculate margins
        df = income.copy()
        df['Gross Margin'] = df['grossProfit'] / df['revenue']
        df['Operating Margin'] = df['operatingIncome'] / df['revenue']
        df['EBITDA Margin'] = df['ebitda'] / df['revenue']
        df['Pre-Tax Margin'] = df['incomeBeforeTax'] / df['revenue']
        df['Net Margin'] = df['netIncome'] / df['revenue']
        # Select only the margin columns and set index to filingDate
        margin_cols = ['Gross Margin', 'Operating Margin', 'EBITDA Margin', 'Pre-Tax Margin', 'Net Margin']
        result = df[margin_cols]
        return result
    
    def get_price_target(self, consensus: bool = True):
        """
        Get the price target for the stock.
        If the stock has gone through a split in the past 3 months, adjust the target accordingly.
        """
        # Get splits and filter for splits in the last 3 months
        splits_df = self.get_stock_splits()
        if not splits_df.empty:
            # Ensure date is datetime
            splits_df['date'] = pd.to_datetime(splits_df['date'])
            three_months_ago = pd.Timestamp.today() - pd.DateOffset(months=3)
            recent_splits = splits_df[splits_df['date'] >= three_months_ago]
        else:
            recent_splits = pd.DataFrame()

        # Get price target
        if consensus:
            target = self.make_request(f"price-target-consensus?")
            if not target or not isinstance(target, list) or not target[0]:
                return target[0] if target else None
            price_target = target[0]
        else:
            target = self.make_request(f"price-target-summary?")
            if not target or not isinstance(target, list) or not target[0]:
                return target
            price_target = target

        # Adjust price target if there was a split in the last 3 months
        if not recent_splits.empty:
            # If multiple splits, apply all sequentially (most recent first)
            for _, split in recent_splits.sort_values('date', ascending=True).iterrows():
                ratio = split['numerator'] / split['denominator']
                if consensus:
                    # Adjust all price target fields that are price values
                    for key in price_target:
                        if isinstance(price_target[key], (int, float)) and price_target[key] > 0:
                            price_target[key] = price_target[key] / ratio
                else:
                    # Adjust all price target fields in the summary
                    for entry in price_target:
                        for key in entry:
                            if isinstance(entry[key], (int, float)) and entry[key] > 0:
                                entry[key] = entry[key] / ratio

        return price_target
        
    def get_stock_splits(self):
        """
        Get the stock splits for the stock
        """
        return pd.DataFrame(self.make_request(f"splits?"))