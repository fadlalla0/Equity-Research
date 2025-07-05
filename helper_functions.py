from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd
import os
import numpy as np
import Stock
from scipy.optimize import minimize
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
from finvizfinance.screener.overview import Overview
import time
from threading import Lock
import streamlit as st
import plotly.graph_objects as go

def make_request(endpoint: str, api_key: str):
    """
    Helper method to make API requests
    """
    base_url = "https://financialmodelingprep.com/stable"
    separator = '&' if '?' in endpoint else '?'
    response = requests.get(f"{base_url}/{endpoint}{separator}apikey={api_key}")
    if response.status_code == 200:
        return response.json()
    return None

def get_directory(directory_type: str, api_key: str):
    """
    Get the directory for a given type
    Available types:
    - company-symbol-list: List of all company symbols
    - financial-statement-symbol-list: List of all financial statement symbols
    - cik-list: List of all CIK numbers
    - symbol-changes-list: List of all symbol changes
    - etf-symbol-search: List of all ETF symbols
    - actively-trading-list: List of all actively trading symbols
    - earnings-transcript-list: List of all earnings transcripts
    - available-exchanges: List of all available exchanges
    - available-sectors: List of all available sectors
    - available-industries: List of all available industries
    - available-countries: List of all available countries
    """
    endpoint = f"{directory_type}?"
    return make_request(endpoint, api_key)

def calculate_ratio(historical_price: pd.DataFrame, metric: pd.DataFrame):
    """
    Calculate the ratio (e.g., P/E) over time by aligning each price date
    with the most recent metric value (e.g., earnings) as of that date.

    Args:
        historical_price (pd.DataFrame): DataFrame with dates as index and price values (single column)
        metric (pd.DataFrame): DataFrame with filingDate as index and metric values (single column)

    Returns:
        pd.Series: Series of calculated ratios indexed by historical_price dates
    """
    # Ensure indices are datetime
    historical_price = historical_price.copy()
    metric = metric.copy()
    historical_price.index = pd.to_datetime(historical_price.index)
    metric.index = pd.to_datetime(metric.index)

    # Remove duplicate indices to avoid ValueError during reindex
    if historical_price.index.has_duplicates:
        historical_price = historical_price[~historical_price.index.duplicated(keep='first')]
    if metric.index.has_duplicates:
        metric = metric[~metric.index.duplicated(keep='first')]

    # Sort both DataFrames by date ascending
    historical_price = historical_price.sort_index()
    metric = metric.sort_index()

    # Forward-fill metric values to all price dates
    # Reindex metric to price dates, using the most recent available metric for each price date
    aligned_metric = metric.reindex(historical_price.index, method='ffill')

    # Get the price and metric as Series
    price_series = historical_price.iloc[:, 0]
    metric_series = aligned_metric.iloc[:, 0]

    # Avoid division by zero
    ratio = price_series.where(metric_series != 0, np.nan) / metric_series.replace(0, np.nan)

    return ratio

def get_prices(symbols: list[str], api_key: str, column: str = 'close', from_date: str = '1970-01-01', to_date: str = datetime.now().strftime('%Y-%m-%d'), update: bool = False):
    """
    Get the prices for a given list of symbols

    Returns a DataFrame with all historical prices for each symbol,
    preserving the full time range for each symbol

    Limits the number of requests to 250 per minute.
    """
    # Create empty dictionary to store DataFrames for each symbol
    price_data = {}

    # Rate limiting variables
    max_requests_per_minute = 250
    request_count = 0
    start_time = time.time()
    lock = Lock()

    # Create a function to process each symbol
    def process_symbol(symbol):
        nonlocal request_count, start_time
        with lock:
            # Check if we need to sleep to respect the rate limit
            if request_count >= max_requests_per_minute:
                elapsed = time.time() - start_time
                if elapsed < 60:
                    time.sleep(60 - elapsed)
                # Reset counter and timer
                request_count = 0
                start_time = time.time()
            request_count += 1

        stock = Stock.Stock(symbol, api_key)
        historical_price = stock.get_historical_price(from_date=from_date, to_date=to_date, update=update)
        historical_price = historical_price.reset_index()

        # Store the date and close price for this symbol
        prices = historical_price[['date', column]].rename(columns={column: symbol})
        return prices.set_index('date')

    # Run all symbol processing in parallel, but chunked to respect rate limit
    results = []
    chunk_size = max_requests_per_minute
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i+chunk_size]
        with ThreadPoolExecutor() as executor:
            chunk_results = list(executor.map(process_symbol, chunk))
        results.extend(chunk_results)
        # If there are more chunks, sleep to respect the rate limit
        if i + chunk_size < len(symbols):
            time.sleep(60)

    # Store results in price_data dictionary
    for symbol, result in zip(symbols, results):
        price_data[symbol] = result

    # Merge all DataFrames on date index, using outer join to preserve all dates
    data = pd.DataFrame()
    for symbol, prices in price_data.items():
        if data.empty:
            data = prices
        else:
            data = data.join(prices, how='outer')

    return data.sort_index(ascending=False)

def get_quotes(symbols: list[str], api_key: str):
    """
    Get the quotes for a given list of symbols
    Limits the number of requests to 250 per minute.
    """
    # Rate limiting variables
    max_requests_per_minute = 250
    request_count = 0
    start_time = time.time()
    lock = Lock()

    # Create a function to process each symbol
    def process_symbol(symbol):
        nonlocal request_count, start_time
        with lock:
            # Check if we need to sleep to respect the rate limit
            if request_count >= max_requests_per_minute:
                elapsed = time.time() - start_time
                if elapsed < 60:
                    time.sleep(60 - elapsed)
                # Reset counter and timer
                request_count = 0
                start_time = time.time()
            request_count += 1

        stock = Stock.Stock(symbol, api_key)
        stock.load_data()
        return stock.get_quote()

    # Run all symbol processing in parallel, but chunked to respect rate limit
    results = []
    chunk_size = max_requests_per_minute
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i+chunk_size]
        with ThreadPoolExecutor() as executor:
            chunk_results = list(executor.map(process_symbol, chunk))
        results.extend(chunk_results)
        # If there are more chunks, sleep to respect the rate limit
        if i + chunk_size < len(symbols):
            time.sleep(60)

    return results

def get_price_targets(symbols: list[str], api_key: str):
    """
    Get the price targets for a given list of symbols
    Limits the number of requests to 250 per minute.
    """
    # Rate limiting variables
    max_requests_per_minute = 250
    request_count = 0
    start_time = time.time()
    lock = Lock()

    # Create a function to process each symbol
    def fetch_price_target(symbol):
        nonlocal request_count, start_time
        with lock:
            # Check if we need to sleep to respect the rate limit
            if request_count >= max_requests_per_minute:
                elapsed = time.time() - start_time
                if elapsed < 60:
                    time.sleep(60 - elapsed)
                # Reset counter and timer
                request_count = 0
                start_time = time.time()
            request_count += 1

        stock = Stock.Stock(symbol, api_key)
        return stock.get_price_target()

    # Run all symbol processing in parallel, but chunked to respect rate limit
    results = []
    chunk_size = max_requests_per_minute
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i+chunk_size]
        with ThreadPoolExecutor() as executor:
            chunk_results = list(executor.map(fetch_price_target, chunk))
        results.extend(chunk_results)
        # If there are more chunks, sleep to respect the rate limit
        if i + chunk_size < len(symbols):
            time.sleep(60)
    return pd.DataFrame(results)

def get_financial_scores(symbols: list[str], api_key: str):
    """
    Get the financial scores for a given list of symbols in parallel
    """
    with ThreadPoolExecutor() as executor:
        # Create a list of futures for each symbol
        futures = [executor.submit(Stock.Stock(symbol, api_key).get_financial_scores) for symbol in symbols]
        # Get results as they complete
        financial_scores = [future.result() for future in futures]
    return pd.concat(financial_scores)

def calculate_historical_ratio(symbol: str, api_key: str):
    """
    Calculate the historical ratio for a given ratio
    ratios:
    - price_to_sales
    - price_to_book
    - price_to_earnings
    - price_to_cash_flow
    - price_to_free_cash_flow
    - price_to_cash
    """
    stock = Stock.Stock(symbol, api_key)

    with ThreadPoolExecutor() as executor:
        historical_price_future = executor.submit(stock.get_historical_price)
        balance_sheet_future = executor.submit(stock.get_statement, 'balance-sheet-statement', 'quarterly')
        income_statement_future = executor.submit(stock.get_statement, 'income-statement', 'quarterly')
        cash_flow_future = executor.submit(stock.get_statement, 'cash-flow-statement', 'quarterly')
        
        historical_price = historical_price_future.result()
        balance_sheet = balance_sheet_future.result()
        income_statement = income_statement_future.result()
        cash_flow = cash_flow_future.result()

    historical_price = historical_price['close']
    historical_price = pd.DataFrame(historical_price)

    balance_sheet = balance_sheet.reset_index()
    balance_sheet = balance_sheet.drop(['fiscalYear', 'period'], axis=1)
    balance_sheet = balance_sheet.set_index('filingDate')

    income_statement = income_statement.reset_index()
    income_statement = income_statement.drop(['fiscalYear', 'period'], axis=1)
    income_statement = income_statement.set_index('filingDate')

    cash_flow = cash_flow.reset_index()
    cash_flow = cash_flow.drop(['fiscalYear', 'period'], axis=1)
    cash_flow = cash_flow.set_index('filingDate')

    # Get common index between all statements
    common_index = balance_sheet.index.intersection(income_statement.index).intersection(cash_flow.index)
    balance_sheet = balance_sheet.loc[common_index]
    income_statement = income_statement.loc[common_index] 
    cash_flow = cash_flow.loc[common_index]

    balance_sheet = balance_sheet.loc[~balance_sheet.index.duplicated(keep='first')]
    income_statement = income_statement.loc[~income_statement.index.duplicated(keep='first')]
    cash_flow = cash_flow.loc[~cash_flow.index.duplicated(keep='first')]

    with ThreadPoolExecutor() as executor:
        net_income_future = executor.submit(stock.calculate_ttm, 'income-statement', 'netIncome')
        shares_future = executor.submit(stock.calculate_ttm, 'income-statement', 'weightedAverageShsOutDil', True)
        revenue_future = executor.submit(stock.calculate_ttm, 'income-statement', 'revenue')
        cash_flow_future = executor.submit(stock.calculate_ttm, 'cash-flow-statement', 'freeCashFlow')

        eps_ttm = net_income_future.result()['netIncome_4q_rolling'] / shares_future.result()['weightedAverageShsOutDil_4q_rolling']
        revenue_ttm = revenue_future.result()['revenue_4q_rolling']
        cash_flow_ttm = cash_flow_future.result()['freeCashFlow_4q_rolling']

    # EPS
    eps_ttm = pd.DataFrame(eps_ttm, columns=['eps_ttm'])

    # book value
    book_value = balance_sheet['totalAssets'] - balance_sheet['totalLiabilities']
    book_value = pd.DataFrame(book_value, columns=['book_value'])
    book_value['book_value'] = book_value['book_value'] / income_statement['weightedAverageShsOut']

    # price to sales
    revenue_ttm = pd.DataFrame(revenue_ttm)
    revenue_ttm = revenue_ttm.loc[common_index]

    price_to_sales = revenue_ttm['revenue_4q_rolling'] / income_statement['weightedAverageShsOut']
    price_to_sales = pd.DataFrame(price_to_sales, columns=['price_to_sales'])

    # Price to free cash flow
    cash_flow_ttm = pd.DataFrame(cash_flow_ttm)
    cash_flow_ttm = cash_flow_ttm.loc[common_index]

    price_to_free_cash_flow = cash_flow_ttm['freeCashFlow_4q_rolling'] / income_statement['weightedAverageShsOut']
    price_to_free_cash_flow = pd.DataFrame(price_to_free_cash_flow, columns=['price_to_free_cash_flow'])

    # Price to cash
    cash_and_cash_eq = balance_sheet['cashAndShortTermInvestments']
    cash_and_cash_eq = pd.DataFrame(cash_and_cash_eq, columns=['cashAndShortTermInvestments'])
    cash_and_cash_eq = cash_and_cash_eq.loc[common_index]

    price_to_cash = cash_and_cash_eq['cashAndShortTermInvestments'] / income_statement['weightedAverageShsOut']
    price_to_cash = pd.DataFrame(price_to_cash, columns=['price_to_cash'])

    # Run ratio calculations in parallel
    with ThreadPoolExecutor() as executor:
        pe_future = executor.submit(calculate_ratio, historical_price, eps_ttm)
        pb_future = executor.submit(calculate_ratio, historical_price, book_value)
        ps_future = executor.submit(calculate_ratio, historical_price, price_to_sales)
        pcf_future = executor.submit(calculate_ratio, historical_price, price_to_free_cash_flow)
        pcash_future = executor.submit(calculate_ratio, historical_price, price_to_cash)

        # Get results
        historical_price['pe'] = pe_future.result()
        historical_price['pb'] = pb_future.result()
        historical_price['ps'] = ps_future.result()
        historical_price['pfcf'] = pcf_future.result()
        historical_price['pc'] = pcash_future.result()

    return historical_price

def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate the beta of an asset relative to a market.
    """
    return asset_returns.cov(market_returns) / market_returns.var()

def calculate_volatility(asset_returns: pd.Series) -> float:
    return asset_returns.std()

def calculate_capm_expected_return(risk_free_rate: float, beta: float, market_return: float) -> float:
    """
    Calculate the expected return of an asset using the Capital Asset Pricing Model (CAPM).

    Args:
        risk_free_rate (float): The risk-free rate of return (as a decimal, e.g., 0.03 for 3%).
        beta (float): The beta of the asset.
        market_return (float): The expected market return (as a decimal, e.g., 0.08 for 8%).

    Returns:
        float: The expected return of the asset according to CAPM.
    """
    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
    return expected_return

def calculate_beta_for_symbol(symbol: str, market_benchmark: str, from_date: datetime, to_date: datetime, time_period: str = 'Monthly') -> float:
    """
    Calculate the beta of a stock symbol relative to a market benchmark over a specified date range and time period.

    Args:
        symbol (str): The stock symbol to calculate beta for.
        market_benchmark (str): The market benchmark symbol.
        from_date (datetime): The start date for the calculation.
        to_date (datetime): The end date for the calculation.
        time_period (str, optional): The time period for resampling ('Daily', 'Monthly', 'Quarterly', 'Yearly'). Defaults to 'Monthly'.

    Returns:
        float: The calculated beta value.
    """
    historical_price = get_prices([symbol, market_benchmark], symbol, from_date=from_date.strftime('%Y-%m-%d'), to_date=to_date.strftime('%Y-%m-%d'))
    if time_period == 'Daily':
        historical_price = historical_price.resample('D').last()
    elif time_period == 'Monthly':
        historical_price = historical_price.resample('ME').last()
    elif time_period == 'Quarterly':
        historical_price = historical_price.resample('Q').last()
    elif time_period == 'Yearly':
        historical_price = historical_price.resample('YE').last()

    historical_price[symbol] = historical_price[symbol].pct_change().dropna()
    historical_price[market_benchmark] = historical_price[market_benchmark].pct_change().dropna()
    beta = calculate_beta(historical_price[symbol], historical_price[market_benchmark])
    expected_return = historical_price[symbol].mean()

    return beta, expected_return

def get_normalized_returns(items: list[str], api_key: str, from_date: str, to_date: str = datetime.now().strftime('%Y-%m-%d')):
    """
    Calculate normalized returns for a list of items (stocks, indices, etc.) starting from a base date.
    All items will start at 0% and show their relative performance over time.

    Args:
        items (list[str]): List of symbols to compare
        api_key (str): API key for fetching historical price data
        from_date (str): Start date in 'YYYY-MM-DD' format
        to_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: DataFrame with normalized returns (in percentage form) for each item
    """
    # Get historical prices
    historical_prices = get_prices(items, api_key, 'close', from_date=from_date, to_date=to_date)
    historical_prices = historical_prices.sort_index()
    historical_prices = historical_prices[(historical_prices.index >= str(from_date)) & (historical_prices.index <= str(to_date))]

    # Normalize so all start at 0% gain
    normalized_returns = historical_prices.divide(historical_prices.iloc[0]) - 1
    normalized_returns *= 100  # Convert to percentage form
    normalized_returns = normalized_returns.reset_index().melt(id_vars=normalized_returns.index.name, var_name='Item', value_name='Return (%)')
    
    return normalized_returns

def finviz_screener(filters: dict):
    screener = Overview()
    screener.set_filter(filters_dict=filters)
    data = screener.screener_view(verbose=0)
    return data

def historical_ratios_for_symbols(ratios: dict[str, pd.DataFrame], ratio_to_plot: str, from_date: str, to_date: str = datetime.now().strftime('%Y-%m-%d')):
    """
    Plot the historical ratios for a given list of symbols
    """

    # Create a DataFrame to hold the data for plotting
    plot_data = pd.DataFrame()

    # Populate the DataFrame with the selected ratio for each symbol
    for symbol, ratios in ratios.items():
        if ratio_to_plot in ratios.columns:
            # For the first symbol, just assign; for others, join
            if plot_data.empty:
                plot_data = ratios[[ratio_to_plot]].rename(columns={ratio_to_plot: symbol})
            else:
                plot_data = plot_data.join(ratios[[ratio_to_plot]].rename(columns={ratio_to_plot: symbol}), how='outer')

    plot_data = plot_data[plot_data.index >= pd.to_datetime(from_date)]
    plot_data = plot_data[plot_data.index <= pd.to_datetime(to_date)]

    return plot_data

def format_number(value):
    """
    Format large numbers with appropriate suffixes (K, M, B, T)
    """
    if abs(value) >= 1e12:
        return f"{value/1e12:.1f}T"
    elif abs(value) >= 1e9:
        return f"{value/1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.1f}K"
    else:
        return f"{value:.1f}"

def sankey_diagram_income_statement(data: dict, title: str):
    """
    Create a Sankey diagram for income statement data with enhanced metrics and visualizations.
    
    Args:
        data (dict): Dictionary containing income statement components and their values
        title (str): Title for the diagram
    
    Returns:
        plotly.graph_objects.Figure: The Sankey diagram figure
    """
    # Calculate key metrics
    revenue = data.get('revenue', 0)
    gross_profit = data.get('grossProfit', 0)
    operating_income = data.get('operatingIncome', 0)
    net_income = data.get('netIncome', 0)
    r_and_d = data.get('researchAndDevelopmentExpenses', 0)
    sg_and_a = data.get('sellingGeneralAndAdministrativeExpenses', 0)
    ebitda = data.get('ebitda', 0)
    interest_expense = data.get('interestExpense', 0)
    income_tax = data.get('incomeTaxExpense', 0)
    
    # Calculate ratios
    gross_margin = (gross_profit / revenue * 100) if revenue else 0
    operating_margin = (operating_income / revenue * 100) if revenue else 0
    net_margin = (net_income / revenue * 100) if revenue else 0
    r_and_d_ratio = (r_and_d / revenue * 100) if revenue else 0
    sg_and_a_ratio = (sg_and_a / revenue * 100) if revenue else 0
    ebitda_margin = (ebitda / revenue * 100) if revenue else 0
    interest_coverage = (operating_income / interest_expense) if interest_expense else 0
    effective_tax_rate = (income_tax / (income_tax + net_income) * 100) if (income_tax + net_income) else 0
    
    # Format metrics for display
    metrics_text = f"""
    • Gross Margin: {gross_margin:.1f}% 
    • Operating Margin: {operating_margin:.1f}% 
    • EBITDA Margin: {ebitda_margin:.1f}% 
    • Net Margin: {net_margin:.1f}% 
    • R&D to Revenue: {r_and_d_ratio:.1f}% 
    • SG&A to Revenue: {sg_and_a_ratio:.1f}% 
    • Interest Coverage: {interest_coverage:.1f}x 
    • Effective Tax Rate: {effective_tax_rate:.1f}%
    """
    
    # Define the nodes (categories) for the Sankey diagram
    node_labels = [
        'Revenue',
        'Cost of Revenue',
        'Gross Profit',
        'Operating Expenses',
        'R&D Expenses',
        'SG&A Expenses',
        'Selling & Marketing',
        'Other Expenses',
        'Operating Income',
        'Interest Income',
        'Interest Expense',
        'Non-Operating Income',
        'Income Before Tax',
        'Income Tax',
        'Net Income'
    ]
    
    nodes = {label: idx for idx, label in enumerate(node_labels)}
    
    # Safely get values or default to 0 if missing
    def get(key): return data.get(key, 0)
    
    # Node values for customdata (sum of incoming links for each node)
    node_values = [
        get('revenue'),
        get('costOfRevenue'),
        get('grossProfit'),
        get('operatingExpenses'),
        get('researchAndDevelopmentExpenses'),
        get('sellingGeneralAndAdministrativeExpenses'),
        get('sellingAndMarketingExpenses'),
        get('otherExpenses'),
        get('operatingIncome'),
        get('interestIncome'),
        get('interestExpense'),
        get('nonOperatingIncomeExcludingInterest'),
        get('incomeBeforeTax'),
        get('incomeTaxExpense'),
        get('netIncome')
    ]
    node_customdata = [format_number(v) for v in node_values]
    
    # Define the links (flows) between nodes with percentage labels
    links = [
        # Revenue flows
        {'source': nodes['Revenue'], 'target': nodes['Cost of Revenue'], 
         'value': get('costOfRevenue'),
         'label': f"{format_number(get('costOfRevenue'))} ({get('costOfRevenue')/revenue*100:.1f}%)" if revenue else "0"},
        {'source': nodes['Revenue'], 'target': nodes['Gross Profit'], 
         'value': get('grossProfit'),
         'label': f"{format_number(get('grossProfit'))} ({get('grossProfit')/revenue*100:.1f}%)" if revenue else "0"},
        
        # Operating expenses flows
        {'source': nodes['Gross Profit'], 'target': nodes['Operating Expenses'], 
         'value': get('operatingExpenses'),
         'label': f"{format_number(get('operatingExpenses'))} ({get('operatingExpenses')/revenue*100:.1f}%)" if revenue else "0"},
        {'source': nodes['Operating Expenses'], 'target': nodes['R&D Expenses'], 
         'value': get('researchAndDevelopmentExpenses'),
         'label': f"{format_number(get('researchAndDevelopmentExpenses'))} ({get('researchAndDevelopmentExpenses')/revenue*100:.1f}%)" if revenue else "0"},
        {'source': nodes['Operating Expenses'], 'target': nodes['SG&A Expenses'], 
         'value': get('sellingGeneralAndAdministrativeExpenses'),
         'label': f"{format_number(get('sellingGeneralAndAdministrativeExpenses'))} ({get('sellingGeneralAndAdministrativeExpenses')/revenue*100:.1f}%)" if revenue else "0"},
        {'source': nodes['Operating Expenses'], 'target': nodes['Selling & Marketing'], 
         'value': get('sellingAndMarketingExpenses'),
         'label': f"{format_number(get('sellingAndMarketingExpenses'))} ({get('sellingAndMarketingExpenses')/revenue*100:.1f}%)" if revenue else "0"},
        {'source': nodes['Operating Expenses'], 'target': nodes['Other Expenses'], 
         'value': get('otherExpenses'),
         'label': f"{format_number(get('otherExpenses'))} ({get('otherExpenses')/revenue*100:.1f}%)" if revenue else "0"},
        
        # Operating income flow
        {'source': nodes['Gross Profit'], 'target': nodes['Operating Income'], 
         'value': get('operatingIncome'),
         'label': f"{format_number(get('operatingIncome'))} ({get('operatingIncome')/revenue*100:.1f}%)" if revenue else "0"},
        
        # Interest flows
        {'source': nodes['Operating Income'], 'target': nodes['Interest Income'], 
         'value': get('interestIncome'),
         'label': f"{format_number(get('interestIncome'))} ({get('interestIncome')/revenue*100:.1f}%)" if revenue else "0"},
        {'source': nodes['Operating Income'], 'target': nodes['Interest Expense'], 
         'value': get('interestExpense'),
         'label': f"{format_number(get('interestExpense'))} ({get('interestExpense')/revenue*100:.1f}%)" if revenue else "0"},
        
        # Non-operating income flow
        {'source': nodes['Operating Income'], 'target': nodes['Non-Operating Income'], 
         'value': get('nonOperatingIncomeExcludingInterest'),
         'label': f"{format_number(get('nonOperatingIncomeExcludingInterest'))} ({get('nonOperatingIncomeExcludingInterest')/revenue*100:.1f}%)" if revenue else "0"},
        
        # Income before tax flow
        {'source': nodes['Operating Income'], 'target': nodes['Income Before Tax'], 
         'value': get('incomeBeforeTax'),
         'label': f"{format_number(get('incomeBeforeTax'))} ({get('incomeBeforeTax')/revenue*100:.1f}%)" if revenue else "0"},
        
        # Tax and net income flows
        {'source': nodes['Income Before Tax'], 'target': nodes['Income Tax'], 
         'value': get('incomeTaxExpense'),
         'label': f"{format_number(get('incomeTaxExpense'))} ({get('incomeTaxExpense')/revenue*100:.1f}%)" if revenue else "0"},
        {'source': nodes['Income Before Tax'], 'target': nodes['Net Income'], 
         'value': get('netIncome'),
         'label': f"{format_number(get('netIncome'))} ({get('netIncome')/revenue*100:.1f}%)" if revenue else "0"}
    ]
    
    # Optional color coding by type
    node_colors = [
        "#00B4D8",  # Revenue - Bright cyan
        "#FF6B6B",  # Cost of Revenue - Coral red
        "#06D6A0",  # Gross Profit - Mint green
        "#FFD166",  # Operating Expenses - Golden yellow
        "#118AB2",  # R&D Expenses - Ocean blue
        "#073B4C",  # SG&A Expenses - Deep navy
        "#7209B7",  # Selling & Marketing - Purple
        "#EF476F",  # Other Expenses - Pink
        "#06D6A0",  # Operating Income - Mint green
        "#3A86FF",  # Interest Income - Royal blue
        "#FF6B6B",  # Interest Expense - Coral red
        "#7209B7",  # Non-Operating Income - Purple
        "#3A86FF",  # Income Before Tax - Royal blue
        "#FF6B6B",  # Income Tax - Coral red
        "#06D6A0"   # Net Income - Mint green
    ]
    
    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="rgba(255,255,255,0.2)", width=0.5),
            label=node_labels,
            color=node_colors,
            customdata=node_customdata,
            hovertemplate="<b>%{label}</b><br>Value: %{customdata}<extra></extra>"
        ),
        link=dict(
            source=[link['source'] for link in links],
            target=[link['target'] for link in links],
            value=[link['value'] for link in links],
            label=[link['label'] for link in links],
            customdata=[format_number(link['value']) for link in links],
            hovertemplate="<b>%{label}</b><br>Value: %{customdata}<br>source: %{source.label}<br>target: %{target.label}<extra></extra>",
            color=[f"rgba{tuple(int(node_colors[link['source']].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.4,)}" for link in links]
        ),
        textfont=dict(
            color="#FFFFFF",  # White text for all nodes
            size=12,
            family="Arial"
        )
    )])
    
    # Update layout with dark theme and metrics
    fig.update_layout(
        title_text=title,
        font=dict(
            size=14,
            color="#E0E0E0",  # Light gray for title
            family="Arial"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=100, l=25, r=25, b=25),  # Increased top margin for metrics
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.9)",
            font_size=12,
            font_family="Arial",
            font_color="#FFFFFF"  # White for hover text
        ),
        annotations=[
            dict(
                x=0.5,
                y=1.1,
                xref="paper",
                yref="paper",
                text=metrics_text,
                showarrow=False,
                font=dict(
                    size=12,
                    color="#E0E0E0",
                    family="Arial"
                ),
                align="center",
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
                borderpad=4
            )
        ]
    )
    
    return fig

def sankey_diagram_balance_sheet(data: dict, title: str):
    """
    Create a Sankey diagram for balance sheet data with enhanced metrics and visualizations.
    Args:
        data (dict): Dictionary containing balance sheet components and their values
        title (str): Title for the diagram
    Returns:
        plotly.graph_objects.Figure: The Sankey diagram figure
    """
    # Node labels
    node_labels = [
        "Cash", "Receivables", "Inventory", "PPE", "Intangibles",
        "Current Assets", "Non-Current Assets", "Total Assets",
        "Current Liabilities", "Long-Term Liabilities", "Total Liabilities",
        "Common Stock", "Retained Earnings", "Total Equity"
    ]
    nodes = {label: idx for idx, label in enumerate(node_labels)}

    # Helper to get value or 0
    def get(key): return data.get(key, 0)

    # Asset nodes
    cash = get('cashAndShortTermInvestments')
    receivables = get('netReceivables') or get('accountsReceivables')
    inventory = get('inventory')
    ppe = get('propertyPlantEquipmentNet')
    intangibles = get('goodwillAndIntangibleAssets') or get('intangibleAssets') or get('goodwill')
    current_assets = get('totalCurrentAssets')
    non_current_assets = get('totalNonCurrentAssets')
    total_assets = get('totalAssets')

    # Liability nodes
    current_liabilities = get('totalCurrentLiabilities')
    long_term_liabilities = get('totalNonCurrentLiabilities')
    total_liabilities = get('totalLiabilities')
    short_term_debt = get('shortTermDebt')
    long_term_debt = get('longTermDebt')
    total_debt = short_term_debt + long_term_debt

    # Equity nodes
    common_stock = get('commonStock')
    retained_earnings = get('retainedEarnings')
    total_equity = get('totalEquity') or get('totalStockholdersEquity')

    # Calculate additional metrics
    working_capital = current_assets - current_liabilities
    net_debt = total_debt - cash
    tangible_assets = total_assets - intangibles
    net_working_capital = (current_assets - cash) - (current_liabilities - short_term_debt)

    # Calculate ratios
    current_ratio = (current_assets / current_liabilities) if current_liabilities else 0
    quick_ratio = ((current_assets - inventory) / current_liabilities) if current_liabilities else 0
    cash_ratio = (cash / current_liabilities) if current_liabilities else 0
    debt_equity = (total_liabilities / total_equity) if total_equity else 0
    equity_ratio = (total_equity / total_assets * 100) if total_assets else 0
    debt_to_assets = (total_liabilities / total_assets * 100) if total_assets else 0
    net_debt_to_equity = (net_debt / total_equity) if total_equity else 0
    working_capital_ratio = (working_capital / total_assets * 100) if total_assets else 0
    tangible_assets_ratio = (tangible_assets / total_assets * 100) if total_assets else 0
    receivables_to_assets = (receivables / total_assets * 100) if total_assets else 0
    inventory_to_assets = (inventory / total_assets * 100) if total_assets else 0
    cash_to_assets = (cash / total_assets * 100) if total_assets else 0

    # Enhanced metrics text with more ratios
    metrics_text = f"""
    • Current Ratio: {current_ratio:.2f}
    • Quick Ratio: {quick_ratio:.2f}
    • Cash Ratio: {cash_ratio:.2f}
    
    • Debt/Equity: {debt_equity:.2f}
    • Net Debt/Equity: {net_debt_to_equity:.2f}
    • Debt to Assets: {debt_to_assets:.1f}%
    
    • Equity Ratio: {equity_ratio:.1f}%
    • Working Capital Ratio: {working_capital_ratio:.1f}%
    • Tangible Assets Ratio: {tangible_assets_ratio:.1f}%
    
    • Cash to Assets: {cash_to_assets:.1f}%
    • Receivables to Assets: {receivables_to_assets:.1f}%
    • Inventory to Assets: {inventory_to_assets:.1f}%

    • Working Capital: {format_number(working_capital)}
    • Net Working Capital: {format_number(net_working_capital)}
    • Net Debt: {format_number(net_debt)}
    """

    # Node values for customdata
    node_values = [
        cash, receivables, inventory, ppe, intangibles,
        current_assets, non_current_assets, total_assets,
        current_liabilities, long_term_liabilities, total_liabilities,
        common_stock, retained_earnings, total_equity
    ]
    node_customdata = [format_number(v) for v in node_values]

    # Links (flows)
    links = [
        # Current assets breakdown
        {"source": nodes["Cash"], "target": nodes["Current Assets"], "value": cash, "label": f"{format_number(cash)}"},
        {"source": nodes["Receivables"], "target": nodes["Current Assets"], "value": receivables, "label": f"{format_number(receivables)}"},
        {"source": nodes["Inventory"], "target": nodes["Current Assets"], "value": inventory, "label": f"{format_number(inventory)}"},
        # Non-current assets breakdown
        {"source": nodes["PPE"], "target": nodes["Non-Current Assets"], "value": ppe, "label": f"{format_number(ppe)}"},
        {"source": nodes["Intangibles"], "target": nodes["Non-Current Assets"], "value": intangibles, "label": f"{format_number(intangibles)}"},
        # Current + Non-current assets to Total Assets
        {"source": nodes["Current Assets"], "target": nodes["Total Assets"], "value": current_assets, "label": f"{format_number(current_assets)}"},
        {"source": nodes["Non-Current Assets"], "target": nodes["Total Assets"], "value": non_current_assets, "label": f"{format_number(non_current_assets)}"},
        # Liabilities
        {"source": nodes["Current Liabilities"], "target": nodes["Total Liabilities"], "value": current_liabilities, "label": f"{format_number(current_liabilities)}"},
        {"source": nodes["Long-Term Liabilities"], "target": nodes["Total Liabilities"], "value": long_term_liabilities, "label": f"{format_number(long_term_liabilities)}"},
        # Equity
        {"source": nodes["Common Stock"], "target": nodes["Total Equity"], "value": common_stock, "label": f"{format_number(common_stock)}"},
        {"source": nodes["Retained Earnings"], "target": nodes["Total Equity"], "value": retained_earnings, "label": f"{format_number(retained_earnings)}"},
        # Total Liabilities + Total Equity to Total Assets
        {"source": nodes["Total Liabilities"], "target": nodes["Total Assets"], "value": total_liabilities, "label": f"{format_number(total_liabilities)}"},
        {"source": nodes["Total Equity"], "target": nodes["Total Assets"], "value": total_equity, "label": f"{format_number(total_equity)}"},
    ]

    # Color palette
    node_colors = [
        "#00B4D8",  # Cash
        "#FFD166",  # Receivables
        "#06D6A0",  # Inventory
        "#3A86FF",  # PPE
        "#7209B7",  # Intangibles
        "#118AB2",  # Current Assets
        "#073B4C",  # Non-Current Assets
        "#2D3142",  # Total Assets
        "#FF6B6B",  # Current Liabilities
        "#EF476F",  # Long-Term Liabilities
        "#D7263D",  # Total Liabilities
        "#FFD166",  # Common Stock
        "#06D6A0",  # Retained Earnings
        "#118AB2",  # Total Equity
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="rgba(255,255,255,0.2)", width=0.5),
            label=node_labels,
            color=node_colors,
            customdata=node_customdata,
            hovertemplate="<b>%{label}</b><br>Value: %{customdata}<extra></extra>"
        ),
        link=dict(
            source=[link['source'] for link in links],
            target=[link['target'] for link in links],
            value=[link['value'] for link in links],
            label=[link['label'] for link in links],
            customdata=[format_number(link['value']) for link in links],
            hovertemplate="<b>%{label}</b><br>Value: %{customdata}<br>source: %{source.label}<br>target: %{target.label}<extra></extra>",
            color=[f"rgba{tuple(int(node_colors[link['source']].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.4,)}" for link in links]
        ),
        textfont=dict(
            color="#FFFFFF",
            size=12,
            family="Arial"
        )
    )])

    fig.update_layout(
        title_text=title,
        font=dict(
            size=14,
            color="#E0E0E0",
            family="Arial"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=150, l=25, r=25, b=25),  # Increased top margin for more metrics
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.9)",
            font_size=12,
            font_family="Arial",
            font_color="#FFFFFF"
        ),
        annotations=[
            dict(
                x=0.5,
                y=1.15,  # Adjusted y position for more metrics
                xref="paper",
                yref="paper",
                text=metrics_text,
                showarrow=False,
                font=dict(
                    size=12,
                    color="#E0E0E0",
                    family="Arial"
                ),
                align="center",
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
                borderpad=4
            )
        ]
    )
    return fig

def sankey_diagram_cashflow(data: dict, title: str):
    """
    Create a Sankey diagram for cash flow statement data with enhanced metrics and visualizations.
    Args:
        data (dict): Dictionary containing cash flow statement components and their values
        title (str): Title for the diagram
    Returns:
        plotly.graph_objects.Figure: The Sankey diagram figure
    """
    # Node labels with more detailed breakdown
    node_labels = [
        # Operating Activities
        "Net Income", "Depreciation", "Stock Compensation", "Deferred Tax",
        "Accounts Receivable", "Inventory", "Accounts Payable", "Other Working Capital",
        "Operating Activities",
        # Investing Activities
        "CapEx", "Acquisitions", "Investments", "Investment Sales", "Other Investing",
        "Investing Activities",
        # Financing Activities
        "Debt Issued", "Debt Repaid", "Stock Issued", "Stock Repurchased",
        "Dividends Paid", "Other Financing",
        "Financing Activities",
        # Net Cash Flow
        "Net Cash Change", "Cash at End"
    ]
    nodes = {label: idx for idx, label in enumerate(node_labels)}

    # Helper to get value or 0
    def get(key): return data.get(key, 0)

    # Operating Activities
    net_income = get('netIncome')
    depreciation = get('depreciationAndAmortization')
    stock_compensation = get('stockBasedCompensation')
    deferred_tax = get('deferredIncomeTax')
    
    # Working Capital Components
    accounts_receivable = get('accountsReceivables')
    inventory = get('inventory')
    accounts_payable = get('accountsPayables')
    other_working_capital = get('otherWorkingCapital')
    working_capital_change = get('changeInWorkingCapital')
    
    operating_cash_flow = get('netCashProvidedByOperatingActivities')

    # Investing Activities
    capex = abs(get('investmentsInPropertyPlantAndEquipment'))
    acquisitions = abs(get('acquisitionsNet'))
    investments = abs(get('purchasesOfInvestments'))
    investment_sales = get('salesMaturitiesOfInvestments')
    other_investing = get('otherInvestingActivities')
    investing_cash_flow = get('netCashProvidedByInvestingActivities')

    # Financing Activities
    debt_issued = get('netDebtIssuance')
    debt_repaid = abs(get('longTermNetDebtIssuance')) if get('longTermNetDebtIssuance') < 0 else 0
    stock_issued = get('netStockIssuance')
    stock_repurchased = abs(get('commonStockRepurchased'))
    dividends_paid = abs(get('netDividendsPaid'))
    other_financing = get('otherFinancingActivities')
    financing_cash_flow = get('netCashProvidedByFinancingActivities')

    # Net Cash Change
    net_cash_change = get('netChangeInCash')
    cash_at_end = get('cashAtEndOfPeriod')
    cash_at_beginning = get('cashAtBeginningOfPeriod')

    # Calculate key metrics
    fcf = operating_cash_flow + investing_cash_flow
    fcf_yield = (fcf / abs(capex) * 100) if capex else 0
    operating_cash_flow_margin = (operating_cash_flow / net_income * 100) if net_income else 0
    capex_to_operating = (capex / operating_cash_flow * 100) if operating_cash_flow else 0
    dividend_payout_ratio = (dividends_paid / net_income * 100) if net_income else 0
    working_capital_ratio = (working_capital_change / operating_cash_flow * 100) if operating_cash_flow else 0
    debt_to_equity = (debt_issued / net_income * 100) if net_income else 0
    stock_repurchase_ratio = (stock_repurchased / net_income * 100) if net_income else 0

    # Enhanced metrics text with better organization
    metrics_text = f"""
    • FCF: {format_number(fcf):>10} 
    • CapEx/OpCF: {capex_to_operating:>5.1f}% 
    • FCF Yield: {fcf_yield:>6.1f}% 
    • Acquisitions: {format_number(acquisitions):>10} 
    • OpCF Margin: {operating_cash_flow_margin:>4.1f}% 
    • Investments: {format_number(investments):>10} 
    • WC Ratio: {working_capital_ratio:>6.1f}% 
    • Debt/Income: {debt_to_equity:>5.1f}% 
    • Net Change: {format_number(net_cash_change):>10} 
    • Div Payout: {dividend_payout_ratio:>5.1f}% 
    • End Cash: {format_number(cash_at_end):>10} 
    • Stock Buyback: {stock_repurchase_ratio:>4.1f}%
    """

    # Node values for customdata
    node_values = [
        net_income, depreciation, stock_compensation, deferred_tax,
        accounts_receivable, inventory, accounts_payable, other_working_capital,
        operating_cash_flow,
        capex, acquisitions, investments, investment_sales, other_investing,
        investing_cash_flow,
        debt_issued, debt_repaid, stock_issued, stock_repurchased,
        dividends_paid, other_financing,
        financing_cash_flow,
        net_cash_change, cash_at_end
    ]
    node_customdata = [format_number(v) for v in node_values]

    # Links (flows)
    links = [
        # Operating Activities
        {"source": nodes["Net Income"], "target": nodes["Operating Activities"], 
         "value": net_income, "label": f"{format_number(net_income)}"},
        {"source": nodes["Depreciation"], "target": nodes["Operating Activities"], 
         "value": depreciation, "label": f"{format_number(depreciation)}"},
        {"source": nodes["Stock Compensation"], "target": nodes["Operating Activities"], 
         "value": stock_compensation, "label": f"{format_number(stock_compensation)}"},
        {"source": nodes["Deferred Tax"], "target": nodes["Operating Activities"], 
         "value": deferred_tax, "label": f"{format_number(deferred_tax)}"},
        {"source": nodes["Accounts Receivable"], "target": nodes["Operating Activities"], 
         "value": accounts_receivable, "label": f"{format_number(accounts_receivable)}"},
        {"source": nodes["Inventory"], "target": nodes["Operating Activities"], 
         "value": inventory, "label": f"{format_number(inventory)}"},
        {"source": nodes["Accounts Payable"], "target": nodes["Operating Activities"], 
         "value": accounts_payable, "label": f"{format_number(accounts_payable)}"},
        {"source": nodes["Other Working Capital"], "target": nodes["Operating Activities"], 
         "value": other_working_capital, "label": f"{format_number(other_working_capital)}"},
        
        # Investing Activities
        {"source": nodes["CapEx"], "target": nodes["Investing Activities"], 
         "value": capex, "label": f"{format_number(capex)}"},
        {"source": nodes["Acquisitions"], "target": nodes["Investing Activities"], 
         "value": acquisitions, "label": f"{format_number(acquisitions)}"},
        {"source": nodes["Investments"], "target": nodes["Investing Activities"], 
         "value": investments, "label": f"{format_number(investments)}"},
        {"source": nodes["Investment Sales"], "target": nodes["Investing Activities"], 
         "value": investment_sales, "label": f"{format_number(investment_sales)}"},
        {"source": nodes["Other Investing"], "target": nodes["Investing Activities"], 
         "value": other_investing, "label": f"{format_number(other_investing)}"},
        
        # Financing Activities
        {"source": nodes["Debt Issued"], "target": nodes["Financing Activities"], 
         "value": debt_issued, "label": f"{format_number(debt_issued)}"},
        {"source": nodes["Debt Repaid"], "target": nodes["Financing Activities"], 
         "value": debt_repaid, "label": f"{format_number(debt_repaid)}"},
        {"source": nodes["Stock Issued"], "target": nodes["Financing Activities"], 
         "value": stock_issued, "label": f"{format_number(stock_issued)}"},
        {"source": nodes["Stock Repurchased"], "target": nodes["Financing Activities"], 
         "value": stock_repurchased, "label": f"{format_number(stock_repurchased)}"},
        {"source": nodes["Dividends Paid"], "target": nodes["Financing Activities"], 
         "value": dividends_paid, "label": f"{format_number(dividends_paid)}"},
        {"source": nodes["Other Financing"], "target": nodes["Financing Activities"], 
         "value": other_financing, "label": f"{format_number(other_financing)}"},
        
        # Net Cash Flow
        {"source": nodes["Operating Activities"], "target": nodes["Net Cash Change"], 
         "value": operating_cash_flow, "label": f"{format_number(operating_cash_flow)}"},
        {"source": nodes["Investing Activities"], "target": nodes["Net Cash Change"], 
         "value": investing_cash_flow, "label": f"{format_number(investing_cash_flow)}"},
        {"source": nodes["Financing Activities"], "target": nodes["Net Cash Change"], 
         "value": financing_cash_flow, "label": f"{format_number(financing_cash_flow)}"},
        
        # Final Cash Position
        {"source": nodes["Net Cash Change"], "target": nodes["Cash at End"], 
         "value": net_cash_change, "label": f"{format_number(net_cash_change)}"}
    ]

    # Color palette with more distinct colors
    node_colors = [
        # Operating Activities
        "#06D6A0",  # Net Income - Mint green
        "#118AB2",  # Depreciation - Ocean blue
        "#FFD166",  # Stock Compensation - Golden yellow
        "#7209B7",  # Deferred Tax - Purple
        "#EF476F",  # Accounts Receivable - Pink
        "#FF6B6B",  # Inventory - Coral red
        "#3A86FF",  # Accounts Payable - Royal blue
        "#00B4D8",  # Other Working Capital - Bright cyan
        "#06D6A0",  # Operating Activities - Mint green
        # Investing Activities
        "#FF6B6B",  # CapEx - Coral red
        "#EF476F",  # Acquisitions - Pink
        "#7209B7",  # Investments - Purple
        "#3A86FF",  # Investment Sales - Royal blue
        "#00B4D8",  # Other Investing - Bright cyan
        "#073B4C",  # Investing Activities - Deep navy
        # Financing Activities
        "#7209B7",  # Debt Issued - Purple
        "#EF476F",  # Debt Repaid - Pink
        "#3A86FF",  # Stock Issued - Royal blue
        "#FF6B6B",  # Stock Repurchased - Coral red
        "#FFD166",  # Dividends Paid - Golden yellow
        "#00B4D8",  # Other Financing - Bright cyan
        "#118AB2",  # Financing Activities - Ocean blue
        # Net Cash Flow
        "#06D6A0",  # Net Cash Change - Mint green
        "#073B4C"   # Cash at End - Deep navy
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="rgba(255,255,255,0.2)", width=0.5),
            label=node_labels,
            color=node_colors,
            customdata=node_customdata,
            hovertemplate="<b>%{label}</b><br>Value: %{customdata}<extra></extra>"
        ),
        link=dict(
            source=[link['source'] for link in links],
            target=[link['target'] for link in links],
            value=[link['value'] for link in links],
            label=[link['label'] for link in links],
            customdata=[format_number(link['value']) for link in links],
            hovertemplate="<b>%{label}</b><br>Value: %{customdata}<br>source: %{source.label}<br>target: %{target.label}<extra></extra>",
            color=[f"rgba{tuple(int(node_colors[link['source']].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.4,)}" for link in links]
        ),
        textfont=dict(
            color="#FFFFFF",
            size=12,
            family="Arial"
        )
    )])

    fig.update_layout(
        title_text=title,
        font=dict(
            size=14,
            color="#E0E0E0",
            family="Arial"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=150, l=25, r=25, b=25),  # Increased top margin for metrics
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.9)",
            font_size=12,
            font_family="Arial",
            font_color="#FFFFFF"
        ),
        annotations=[
            dict(
                x=0.5,
                y=1.15,  # Adjusted y position for metrics
                xref="paper",
                yref="paper",
                text=metrics_text,
                showarrow=False,
                font=dict(
                    size=12,
                    color="#E0E0E0",
                    family="Arial"
                ),
                align="center",
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
                borderpad=4
            )
        ]
    )
    return fig
