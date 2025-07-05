import streamlit as st
import pandas as pd
import Stock
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor
from helper_functions import *
from datetime import datetime
import os
import plotly.graph_objects as go
import urllib.parse

@st.cache_data
def load_main_data(data: pd.DataFrame, api_key: str):
    """
    Loads the main data from a CSV file in parallel.
    
    Args:
        file_name (str, optional): The name of the CSV file to load. Defaults to 'portfolio.csv'.
        
    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The processed data with additional columns
            - dict: A dictionary mapping symbols to Stock objects
    """
    symbols = data['symbol'].unique()

    def get_stock_info(symbol):
        stock = Stock.Stock(symbol, api_key)
        stock.load_data()
        return {
            'symbol': symbol,
            'stock': stock,
            'sector': stock.sector,
            'industry': stock.industry,
            'price': stock.price,
            'eps': stock.eps,
            'change': stock.change
        }

    results = []
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(get_stock_info, symbols))

    # Create the info dict and lists
    info = {res['symbol']: res['stock'] for res in results}
    sector = [res['sector'] for res in results]
    industry = [res['industry'] for res in results]
    price = [res['price'] for res in results]
    eps = [res['eps'] for res in results]
    change = [res['change'] for res in results]

    data = data.set_index('symbol')
    data['sector'] = sector
    data['industry'] = industry
    data['price'] = price
    data['value'] = data['shares'] * data['price']
    data['eps'] = eps
    data['pe'] = data['price'] / data['eps']
    data['change'] = [chg * sh for chg, sh in zip(change, data['shares'])]
    data = data.drop('shares', axis=1)

    return data, info

@st.cache_data
def treasury_yield(from_date: str, to_date: str, api_key: str):
    """
    Get treasury yield for a given stock symbol.
    """
    return make_request(f'treasury-rates?from={from_date}&to={to_date}&', api_key)

@st.cache_data
def get_key_metrics(symbol: str, api_key: str, ttm: bool = True):
    """
    Get key metrics for a given stock symbol.
    
    Args:
        symbol (str): The stock symbol to get metrics for
        ttm (bool, optional): If True, get trailing twelve months metrics. Defaults to True.
        
    Returns:
        pd.DataFrame: DataFrame containing the key metrics
    """
    return Stock.Stock(symbol, api_key).get_key_metrics(ttm)

@st.cache_data
def get_ratios(symbol: str, api_key: str, ttm: bool = True):
    """
    Get financial ratios for a given stock symbol.
    
    Args:
        symbol (str): The stock symbol to get ratios for
        ttm (bool, optional): If True, get trailing twelve months ratios. Defaults to True.
        
    Returns:
        pd.DataFrame: DataFrame containing the financial ratios
    """
    return Stock.Stock(symbol, api_key).get_ratios(ttm)

@st.cache_data
def get_statement(symbol: str, api_key: str, statement: str, period: str = 'annual', growth: bool = False):
    """
    Get financial statements for a given stock symbol.
    
    Args:
        symbol (str): The stock symbol to get statements for
        statement (str): Type of statement ('income-statement', 'balance-sheet-statement', 'cash-flow-statement')
        period (str, optional): Period of the statement ('annual' or 'quarter'). Defaults to 'annual'.
        growth (bool, optional): If True, get growth rates. Defaults to False.
        
    Returns:
        pd.DataFrame: DataFrame containing the financial statement data
    """
    df = Stock.Stock(symbol, api_key).get_statement(statement, period, growth)
    if period == 'annual':
        # Drop the filingDate index level
        df = df.reset_index().drop('filingDate', axis=1).set_index(['fiscalYear', 'period'])
    
    return df

@st.cache_data
def get_revenue_segmentation(symbol: str, api_key: str, product: bool = True):
    """
    Get revenue segmentation for a given stock symbol.
    
    Args:
        symbol (str): The stock symbol to get revenue segmentation for
        product (bool, optional): If True, get product revenue segmentation. Defaults to True.
        
    Returns:
        pd.DataFrame: DataFrame containing the revenue segmentation data
    """
    return Stock.Stock(symbol, api_key).get_revenue_segmentation(product)

@st.cache_data
def get_historical_price_ratio(symbol: str, api_key: str):
    """
    Get historical price ratio for a given stock symbol.
    """
    return calculate_historical_ratio(symbol, api_key)

@st.cache_data
def get_financial_scores_from_symbols(symbols: list[str], api_key: str):
    """
    Get financial scores for a given list of stock symbols.
    """
    return get_financial_scores(symbols, api_key)

@st.cache_data
def get_analyst_estimates_from_symbol(symbol: str, api_key: str):
    """
    Get analyst estimates for a given list of stock symbols.
    """
    return Stock.Stock(symbol, api_key).get_analyst_estimates()

@st.cache_data
def get_margins_from_symbol(symbol: str, api_key: str, period: str = 'annual'):
    """
    Get margins for a given stock symbol.
    """
    return Stock.Stock(symbol, api_key).calculate_margins(period)

@st.cache_data
def get_economic_indicators(name: str, from_date: str, to_date: str, api_key: str):
    """
    Get economic indicators data for a given indicator name and date range.
    
    Args:
        name (str): The name of the economic indicator (e.g., 'GDP', 'CPI', 'inflationRate')
        from_date (str): Start date in 'YYYY-MM-DD' format
        to_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        dict: JSON response containing the economic indicator data
    """
    endpoint = f"economic-indicators?name={name}&from={from_date}&to={to_date}&"
    return make_request(endpoint, api_key)

@st.cache_data
def get_directory_cached(directory_type: str, api_key: str):
    """
    Get directory data for a given type.
    """
    return get_directory(directory_type, api_key)

@st.cache_data
def get_historical_sector_pe(sector: str, exchange: str, from_date: str, to_date: str, api_key: str):
    """
    Get historical sector PE ratio data.
    
    Args:
        sector (str): The sector name (e.g., 'Energy', 'Technology')
        exchange (str): The exchange name (e.g., 'NASDAQ', 'NYSE')
        from_date (str): Start date in 'YYYY-MM-DD' format
        to_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        dict: JSON response containing the historical sector PE data
    """
    # URL encode the sector name to handle spaces and special characters
    encoded_sector = urllib.parse.quote(sector)
    endpoint = f"historical-sector-pe?exchange={exchange}&sector={encoded_sector}&from={from_date}&to={to_date}&"
    
    response = make_request(endpoint, api_key)
    
    if not response:
        st.error(f"No data received from API for sector {sector}")
        return None
        
    # Validate response format
    if not isinstance(response, list):
        st.error(f"Invalid response format for sector {sector}")
        return None
        
    # Convert from_date and to_date to datetime for comparison
    try:
        from_date_dt = pd.to_datetime(from_date)
        to_date_dt = pd.to_datetime(to_date)
        current_date = pd.Timestamp.now()
    except:
        st.error("Invalid date format in from_date or to_date")
        return None
        
    # Clean and validate the data
    cleaned_data = []
    for item in response:
        if not isinstance(item, dict):
            continue
            
        # Ensure required fields are present
        required_fields = ['date', 'sector', 'exchange', 'pe']
        if not all(field in item for field in required_fields):
            continue
            
        # Validate date is within requested range and not in the future
        try:
            item_date = pd.to_datetime(item['date'])
            if item_date > current_date or item_date < from_date_dt or item_date > to_date_dt:
                continue
        except:
            continue
            
        # Remove duplicate exchange key if present
        if 'exchange' in item and isinstance(item['exchange'], list):
            item['exchange'] = item['exchange'][0]
            
        cleaned_data.append(item)
    
    if not cleaned_data:
        st.warning(f"No valid data found for sector {sector} within the specified date range")
        return None
        
    # Sort data by date
    cleaned_data.sort(key=lambda x: pd.to_datetime(x['date']))
        
    return {'data': cleaned_data}

@st.cache_data
def get_historical_industry_pe(industry: str, exchange: str, from_date: str, to_date: str, api_key: str):
    """
    Get historical industry PE ratio data.
    
    Args:
        industry (str): The industry name (e.g., 'Software', 'Banks')
        exchange (str): The exchange name (e.g., 'NASDAQ', 'NYSE')
        from_date (str): Start date in 'YYYY-MM-DD' format
        to_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        dict: JSON response containing the historical industry PE data
    """
    # URL encode the industry name to handle spaces and special characters
    encoded_industry = urllib.parse.quote(industry)
    endpoint = f"historical-industry-pe?industry={encoded_industry}&exchange={exchange}&from={from_date}&to={to_date}&"
    
    response = make_request(endpoint, api_key)
    
    if not response:
        st.error(f"No data received from API for industry {industry}")
        return None
        
    # Validate response format
    if not isinstance(response, list):
        st.error(f"Invalid response format for industry {industry}")
        return None
        
    # Convert from_date and to_date to datetime for comparison
    try:
        from_date_dt = pd.to_datetime(from_date)
        to_date_dt = pd.to_datetime(to_date)
        current_date = pd.Timestamp.now()
    except:
        st.error("Invalid date format in from_date or to_date")
        return None
        
    # Clean and validate the data
    cleaned_data = []
    for item in response:
        if not isinstance(item, dict):
            continue
            
        # Ensure required fields are present
        required_fields = ['date', 'industry', 'exchange', 'pe']
        if not all(field in item for field in required_fields):
            continue
            
        # Validate date is within requested range and not in the future
        try:
            item_date = pd.to_datetime(item['date'])
            if item_date > current_date or item_date < from_date_dt or item_date > to_date_dt:
                continue
        except:
            continue
            
        # Remove duplicate exchange key if present
        if 'exchange' in item and isinstance(item['exchange'], list):
            item['exchange'] = item['exchange'][0]
            
        cleaned_data.append(item)
    
    if not cleaned_data:
        st.warning(f"No valid data found for industry {industry} within the specified date range")
        return None
        
    # Sort data by date
    cleaned_data.sort(key=lambda x: pd.to_datetime(x['date']))
        
    return {'data': cleaned_data}

def get_prices_from_symbols(symbols: list[str], api_key: str, column: str = 'close', from_date: str = '1970-01-01', to_date: str = datetime.now().strftime('%Y-%m-%d'), update: bool = False):
    """
    Get historical prices for a given list of stock symbols.
    """
    return get_prices(symbols, api_key, column, from_date, to_date, update)

@st.cache_data
def get_historical_price(symbol: str, api_key: str, update: bool = False):
    """
    Get historical price data for a given stock symbol.
    
    Args:
        symbol (str): The stock symbol to get historical prices for
        
    Returns:
        pd.DataFrame: DataFrame containing the historical price data
    """
    return Stock.Stock(symbol, api_key).get_historical_price(update=update)

def get_key_data(data: pd.DataFrame, api_key: str):
    """
    Fetch key metrics and ratios for all symbols in the data.
    
    Args:
        data (pd.DataFrame): DataFrame containing stock symbols
        
    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Key metrics for all symbols
            - pd.DataFrame: Financial ratios for all symbols
    """
    symbols = data.index.tolist()

    def fetch_data(symbol):
        return {
            'symbol': symbol,
            'key_metrics': get_key_metrics(symbol, api_key),
            'ratios': get_ratios(symbol, api_key)
        }

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_data, symbols))

    key_metrics = pd.concat([res['key_metrics'] for res in results])
    ratios = pd.concat([res['ratios'] for res in results])

    return key_metrics, ratios

def basic_metrics(data: pd.DataFrame):
    """
    Display basic portfolio metrics and visualizations.
    
    Args:
        data (pd.DataFrame): DataFrame containing portfolio data with required columns:
            - value: Portfolio value
            - change: Price change
            - sector: Stock sector
            - industry: Stock industry
            - pe: Price-to-earnings ratio
    """
    st.header('Basic Metrics')

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Portfolio Value", "${:,.2f}".format(data['value'].sum()), f"{data['change'].sum(): 0.2f}")
        fig1 = px.pie(data, values='value', names='sector', title='Weight by Sector')
        st.plotly_chart(fig1)

    with col2:
        portfolio_pe = data['value'] / data['value'].sum() * data['pe']
        st.metric("Portfolio PE Ratio", "{:,.2f}".format(portfolio_pe.sum()))
        fig2 = px.pie(data, values='value', names='industry', title='Weight by Industry')
        st.plotly_chart(fig2)
    fig3 = px.pie(data, values='value', names=data.index, title='Weight by Industry')
    st.plotly_chart(fig3)
    st.dataframe(data)

def display_financial_statements(symbol: str, api_key: str):
    """
    Display financial statements and metrics for a given stock symbol concurrently.
    
    Args:
        symbol (str): The stock symbol to display financial data for
        
    Displays:
        - Income Statement
        - Balance Sheet
        - Cash Flow Statement
        - Key Metrics
        - Financial Ratios
    """
    with st.container():
        st.markdown("### 🗓️ Reporting Period")
        period = st.radio(
            "Choose between annual and quarterly financial statements:",
            options=["Annual 📅", "Quarterly 📆"],
            index=0,
            horizontal=True,
            help="Select the type of financial statement period you want to display."
        )
        period = "annual" if "Annual" in period else "quarter"
        
        st.divider()

        # Fetch all data concurrently
        def fetch_data():
            with ThreadPoolExecutor() as executor:
                futures = {
                    'income': executor.submit(get_statement, symbol, api_key, "income-statement", period),
                    'balance': executor.submit(get_statement, symbol, api_key, "balance-sheet-statement", period),
                    'cashflow': executor.submit(get_statement, symbol, api_key, "cash-flow-statement", period),
                    'metrics': executor.submit(get_key_metrics, symbol, api_key, False),
                    'ratios': executor.submit(get_ratios, symbol, api_key, False),
                    'ttm_metrics': executor.submit(get_key_metrics, symbol, api_key, True),
                    'ttm_ratios': executor.submit(get_ratios, symbol, api_key, True),
                    'product_segmentation': executor.submit(get_revenue_segmentation, symbol, api_key, True),
                    'geographic_segmentation': executor.submit(get_revenue_segmentation, symbol, api_key, False),
                    'margins': executor.submit(get_margins_from_symbol, symbol, api_key, period)
                }
                return {key: future.result() for key, future in futures.items()}

        # Get all data concurrently
        data = fetch_data()

        st.header("Sankey Diagrams")
        for statement_type, title, key in [
            ('income', "Income Statement Sankey Diagram", 'income'),
            ('balance', "Balance Sheet Sankey Diagram", 'balance'),
            ('cashflow', "Cash Flow Statement Sankey Diagram", 'cashflow')
        ]:
            with st.expander(f"🔄 {title}", expanded=False):
                st.subheader(title)
                if period == 'annual':
                    available_periods = data[statement_type].index.get_level_values('fiscalYear').unique()
                    selected_period = st.selectbox("Select Year", available_periods, key=f'{key}_annual')
                    statement_data = data[statement_type].xs(selected_period, level='fiscalYear')
                    sankey_data = {k: v['FY'] for k, v in statement_data.to_dict().items()}
                else:
                    available_periods = data[statement_type].index.unique()
                    period_mapping = {f"{year} {quarter}": (year, quarter, date) 
                                      for year, quarter, date in available_periods}
                    selected_period_formatted = st.selectbox("Select Year and Quarter", 
                                                             list(period_mapping.keys()), key=f'{key}_quarterly')
                    selected_period = period_mapping[selected_period_formatted]
                    statement_data = data[statement_type].loc[selected_period]
                    sankey_data = statement_data.to_dict()

                display_sankey_diagram(sankey_data, f"{title} Flow for {selected_period_formatted if period != 'annual' else selected_period}", key)
        st.divider()
        # Create combined metrics list with prefixes
        # Combine all metrics into a single list
        all_metrics = [
            f"{prefix}: {col}"
            for prefix, df in [
                ('IS', data['income']),
                ('BS', data['balance']),
                ('CF', data['cashflow']),
            ] if df is not None
            for col in df.columns
        ]

        # INSERT_YOUR_REWRITE_HERE
        st.header("Combined Metrics Analysis")
        with st.expander("📈 Combined Metrics Analysis", expanded=False):
            # Multi-select for all metrics
            selected_metrics = st.multiselect(
                "Select metrics to display",
                all_metrics,
                help="Choose metrics from different statements to compare trends"
            )

            if selected_metrics:
                # Create a combined DataFrame for plotting
                plot_data = pd.DataFrame()
                x_axis = 'fiscalYear' if period == 'annual' else 'filingDate'
                df_map = {
                    'IS': data['income'],
                    'BS': data['balance'],
                    'CF': data['cashflow'],
                }
                
                for metric in selected_metrics:
                    prefix, col = metric.split(': ', 1)
                    source_df = df_map[prefix]
                    temp_df = source_df.reset_index()[[x_axis, col]].rename(columns={col: metric})
                    plot_data = temp_df if plot_data.empty else plot_data.merge(temp_df, on=x_axis, how='outer')

                # Create the line chart with markers (dots)
                fig = px.line(
                    plot_data,
                    x=x_axis,
                    y=selected_metrics,
                    title='Financial Metrics Trends',
                    labels={'value': 'Amount', x_axis: 'Year' if period == 'annual' else 'Date'},
                    markers=True
                )
                st.plotly_chart(fig)
                
                plot_metric_percentage_change(plot_data, x_axis, selected_metrics)
            
        st.divider()
        st.header("Financial Statements Analysis")
        # Function to create subheader, multiselect, and plot for each financial statement
        def create_financial_section(title, data_key, default_count=5, chart_count=3):
            with st.expander(title, expanded=False):
                columns = st.multiselect(
                    f"Select {title} Columns", 
                    data[data_key].columns.tolist(), 
                    default=data[data_key].columns[:default_count].tolist()
                )
                st.dataframe(data[data_key][columns])
                
                chart_cols = st.multiselect(
                    f"Select metrics for {title} Chart",
                    data[data_key].columns.tolist(),
                    default=data[data_key].columns[:chart_count].tolist()
                )
                if chart_cols:
                    df = data[data_key].reset_index()
                    # Key metrics and ratios only have annual data
                    if data_key in ['metrics', 'ratios']:
                        x_axis = 'fiscalYear'
                    else:
                        x_axis = 'fiscalYear' if period == 'annual' else 'filingDate'
                    fig = px.line(
                        df,
                        x=x_axis,
                        y=chart_cols,
                        title=f'{title} Trends',
                        labels={'value': 'Amount', x_axis: 'Year' if x_axis == 'fiscalYear' else 'Date'},
                        markers=True
                    )
                    st.plotly_chart(fig)


        # Create sections for each financial statement
        create_financial_section("📊 Income Statement", 'income')
        create_financial_section("📋 Balance Sheet", 'balance')
        create_financial_section("💸 Cash Flow Statement", 'cashflow')
        create_financial_section("📌 Key Metrics", 'metrics')
        create_financial_section("📐 Financial Ratios", 'ratios')
        
        # TTM sections with bar charts
        with st.expander("📌 Key Metrics TTM", expanded=False):
            create_bar_chart(data['ttm_metrics'], 'Key Metrics TTM')

        with st.expander("📐 Financial Ratios TTM", expanded=False):
            create_bar_chart(data['ttm_ratios'], 'Financial Ratios TTM')

        st.divider()

        st.header("Segmentation Analysis")
        # Product Segmentation
        with st.expander("💰 Product Segmentation", expanded=False):
            fig_product = px.bar(
                data['product_segmentation'],
                x='date',
                y='revenue',
                color='segment',
                title='Revenue by Product Segment',
                labels={'revenue': 'Revenue', 'date': 'Date', 'segment': 'Product Segment'}
            )
            st.plotly_chart(fig_product)

            available_dates = data['product_segmentation']['date'].drop_duplicates().sort_values(ascending=False)
            selected_date = st.selectbox(
                "Select Date for Product Segmentation Pie Chart",
                available_dates,
                index=0
            )
            pie_data = data['product_segmentation'][data['product_segmentation']['date'] == selected_date]
            fig_product_pie = px.pie(
                pie_data,
                names='segment',
                values='revenue',
                title=f"Product Segmentation Pie Chart ({selected_date})",
                labels={'revenue': 'Revenue', 'segment': 'Product Segment'}
            )
            st.plotly_chart(fig_product_pie)

        # Geographic Segmentation  
        with st.expander("💰 Geographic Segmentation", expanded=False):
            fig_geo = px.bar(
                data['geographic_segmentation'],
                x='date', 
                y='revenue',
                color='segment',
                title='Revenue by Geographic Region',
                labels={'revenue': 'Revenue', 'date': 'Date', 'segment': 'Geographic Region'}
            )
            st.plotly_chart(fig_geo)
        
        st.header("📊 Margins Analysis")
        # Margins
        with st.expander("📊 Margins", expanded=False):
            margins = data['margins']
            st.dataframe(margins)
            x_axis = 'fiscalYear' if period == 'annual' else 'filingDate'
            columns = ['Gross Margin', 'Operating Margin', 'EBITDA Margin', 'Pre-Tax Margin', 'Net Margin']
            df_margins = margins.reset_index()
            fig_margins = px.line(
                df_margins,
                x=x_axis,
                y=columns,
                title='Margins',
                labels={'value': 'Amount', x_axis: 'Year' if period == 'annual' else 'Date'},
                markers=True
            )
            st.plotly_chart(fig_margins)

def display_sankey_diagram(data: dict, title: str, type: str):
    """
    Display a Sankey diagram for income statement data.
    
    Args:
        data (dict): Dictionary containing income statement components and their values
        title (str): Title for the diagram
    """
    st.header(title)
    if type == 'income':
        fig = sankey_diagram_income_statement(data, title)
    elif type == 'balance':
        fig = sankey_diagram_balance_sheet(data, title)
    elif type == 'cashflow':
        fig = sankey_diagram_cashflow(data, title)
    st.plotly_chart(fig)

def display_financial_scores(scores_df):
    """Display financial scores in a clean, organized format"""
    if scores_df is None or scores_df.empty:
        return
    
    # Create three columns for the metrics
    col1, col2, col3 = st.columns(3)
    
    # Format large numbers for better readability
    def format_number(num):
        if abs(num) >= 1e12:
            return f"${num/1e12:.2f}T"
        elif abs(num) >= 1e9:
            return f"${num/1e9:.2f}B"
        elif abs(num) >= 1e6:
            return f"${num/1e6:.2f}M"
        return f"${num:,.2f}"
    
    # Display in first column
    with col1:
        st.metric("Altman Z-Score", f"{scores_df['altmanZScore'].iloc[0]:.2f}")
        st.metric("Piotroski Score", f"{scores_df['piotroskiScore'].iloc[0]}/9")
        st.metric("Working Capital", format_number(scores_df['workingCapital'].iloc[0]))
    
    # Display in second column
    with col2:
        st.metric("Total Assets", format_number(scores_df['totalAssets'].iloc[0]))
        st.metric("Retained Earnings", format_number(scores_df['retainedEarnings'].iloc[0]))
        st.metric("EBIT", format_number(scores_df['ebit'].iloc[0]))
    
    # Display in third column
    with col3:
        st.metric("Market Cap", format_number(scores_df['marketCap'].iloc[0]))
        st.metric("Total Liabilities", format_number(scores_df['totalLiabilities'].iloc[0]))
        st.metric("Revenue", format_number(scores_df['revenue'].iloc[0]))

def display_stock_checklist(symbol: str, api_key: str):
    """
    Display an interactive stock analysis checklist based on Peter Lynch's investment principles.
    
    Args:
        symbol (str): The stock symbol to save the checklist for
        
    Returns:
        dict: A dictionary containing all checklist items and their values
    """
    checklist = {}

    # Define file path
    file_path = f"checklist/{symbol}.csv"
    previous_values = {}

    # If file exists, load the most recent row's values for each field
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path)
            if not existing_df.empty:
                last_row = existing_df.iloc[-1]
                previous_values = last_row.to_dict()
        except Exception as e:
            st.warning(f"Could not load previous checklist: {e}")

    # Stock Category Selection
    checklist["stock category"] = st.selectbox(
        "1. Stock Category",
        ["", "slow grower", "stalwart", "fast grower", "turnaround", "asset play", "cyclical"],
        index=(["", "slow grower", "stalwart", "fast grower", "turnaround", "asset play", "cyclical"].index(previous_values.get("stock category", "")) if previous_values.get("stock category", "") in ["", "slow grower", "stalwart", "fast grower", "turnaround", "asset play", "cyclical"] else 0)
    )

    # Financial Metrics
    checklist["percent of sales"] = st.text_area(
        "2. Percent of Sales", 
        value=previous_values.get("percent of sales", "")
    )
    checklist["price/earnings"] = st.text_area(
        "3. Price/Earnings", 
        value=previous_values.get("price/earnings", "")
    )
    checklist["cash position"] = st.text_area(
        "4. Cash Position", 
        value=previous_values.get("cash position", "")
    )
    checklist["debt factor"] = st.text_area(
        "5. Debt Factor", 
        value=previous_values.get("debt factor", "")
    )
    checklist["dividends"] = st.text_area(
        "6. Dividends", 
        value=previous_values.get("dividends", "")
    )
    checklist["does it pay?"] = st.text_area(
        "7. Does it pay?", 
        value=previous_values.get("does it pay?", "")
    )
    checklist["book value"] = st.text_area(
        "8. Book Value", 
        value=previous_values.get("book value", "")
    )
    checklist["more hidden assets"] = st.text_area(
        "9. More Hidden Assets", 
        value=previous_values.get("more hidden assets", "")
    )
    checklist["cash flow"] = st.text_area(
        "10. Cash Flow", 
        value=previous_values.get("cash flow", "")
    )

    # Management and Operations
    checklist["inventories"] = st.text_area(
        "11. Inventories (management's discussion of earnings)", 
        value=previous_values.get("inventories", "")
    )
    checklist["pension plans"] = st.text_area(
        "12. Pension Plans", 
        value=previous_values.get("pension plans", "")
    )
    checklist["growth rate"] = st.text_area(
        "13. Growth Rate", 
        value=previous_values.get("growth rate", "")
    )

    # Final Analysis
    checklist["final notes"] = st.text_area(
        "14. Final Notes", 
        value=previous_values.get("final notes", ""), 
        height=150
    )

    # Add timestamp
    checklist["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create submit button
    if st.button("Save Checklist"):
        # Create checklist directory if it doesn't exist
        os.makedirs("checklist", exist_ok=True)
        
        # Convert checklist to DataFrame
        df = pd.DataFrame([checklist])
        
        # Check if file exists to append or create new
        if os.path.exists(file_path):
            try:
                existing_df = pd.read_csv(file_path)
                updated_df = pd.concat([existing_df, df], ignore_index=True)
                updated_df.to_csv(file_path, index=False)
                st.success(f"Checklist updated for {symbol}")
            except Exception as e:
                st.error(f"Failed to update checklist: {e}")
        else:
            try:
                df.to_csv(file_path, index=False)
                st.success(f"New checklist created for {symbol}")
            except Exception as e:
                st.error(f"Failed to create checklist: {e}")
    
    return checklist

def display_compounded_returns_chart(api_key: str):
    """
    Display a compounded returns chart for the given items over a specified date range.

    Args:
        api_key (str): The API key for financial data.
    """
    st.header('Compounded Returns Chart')
    spdr_sector_etfs = {
        "XLC": "Communication Services",  # Communication Services
        "XLY": "Consumer Discretionary",  # Consumer Discretionary
        "XLP": "Consumer Staples",  # Consumer Staples
        "XLE": "Energy",  # Energy
        "XLF": "Financials",  # Financials
        "XLV": "Health Care",  # Health Care
        "XLI": "Industrials",  # Industrials
        "XLB": "Materials",  # Materials
        "XLRE": "Real Estate", # Real Estate
        "XLK": "Technology",  # Technology
        "XLU": "Utilities"   # Utilities
    }
    spdr_items = [x[0] for x in spdr_sector_etfs.items()]
    selected_items = st.text_input("Enter items separated by commas:")
    items = [item.strip() for item in selected_items.split(",") if item.strip()]
    col1, col2 = st.columns(2)
    from_date = col1.date_input("From Date", value=datetime(datetime.now().year - 5, datetime.now().month, datetime.now().day).date(), min_value=datetime(1970, 1, 1), max_value=datetime.today().date())
    to_date = col2.date_input("To Date", value=datetime.today().date(), min_value=datetime(1970, 1, 1), max_value=datetime.today().date())
    from_date = from_date.strftime('%Y-%m-%d')
    to_date = to_date.strftime('%Y-%m-%d')
    if items:
        normalized_returns = get_normalized_returns(items, api_key, from_date, to_date)
        # Plot
        fig = px.line(normalized_returns, x=normalized_returns.columns[0], y='Return (%)', color='Item',
                    title='Relative Performance from Initial Price (Start = 0%)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        normalized_returns = get_normalized_returns(spdr_items, api_key, from_date, to_date)
        fig = px.line(normalized_returns, x=normalized_returns.columns[0], y='Return (%)', color='Item',
                    title='Relative Performance from Initial Price (Start = 0%)')
        st.plotly_chart(fig, use_container_width=True)

def display_treasury_yield_chart(data, selected_rates=None):
    """
    Display treasury yield data in a line chart.
    
    Args:
        data (dict or list): Treasury yield data from the API
        selected_rates (list, optional): List of selected rate types to display
    """
    if isinstance(data, dict) and 'data' in data:
        df = pd.DataFrame(data['data'])
    else:
        df = pd.DataFrame(data)

    if not df.empty:
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Filter columns based on selected rates if provided
            if selected_rates:
                # Map our rate codes to the actual column names in the data
                rate_mapping = {
                    '1month': 'month1',
                    '2month': 'month2',
                    '3month': 'month3',
                    '6month': 'month6',
                    '1year': 'year1',
                    '2year': 'year2',
                    '3year': 'year3',
                    '5year': 'year5',
                    '7year': 'year7',
                    '10year': 'year10',
                    '20year': 'year20',
                    '30year': 'year30'
                }
                
                # Convert our rate codes to the actual column names
                selected_columns = [rate_mapping.get(rate, rate) for rate in selected_rates]
                available_rates = [col for col in df.columns if col != 'date']
                selected_columns = [col for col in selected_columns if col in available_rates]
                
                if not selected_columns:
                    st.warning("None of the selected rates are available in the data.")
                    return
                
                df = df[['date'] + selected_columns]
            
            # Create the line chart
            fig = px.line(
                df,
                x='date',
                y=df.columns[1:],  # All columns except 'date'
                labels={
                    'value': 'Yield (%)',
                    'date': 'Date'
                },
                title="Treasury Yield History"
            )
            
            # Add hover template for better data display
            fig.update_traces(
                hovertemplate="<br>".join([
                    "Date: %{x}",
                    "Yield: %{y:.2f}%",
                    "<extra></extra>"
                ])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the raw data
            st.subheader("Raw Data")
            st.dataframe(df)
    else:
        st.warning("No treasury yield data available for the selected period.")

def display_economic_indicator_chart(data):
    """
    Display economic indicator data in a line chart.
    
    Args:
        data (dict or list): Economic indicator data from the API
    """
    if isinstance(data, dict) and 'data' in data:
        df = data['data']
    else:
        df = data

    if isinstance(df, list):
        import pandas as pd
        df = pd.DataFrame(df)

    if not df.empty:
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Create the line chart
            fig = px.line(
                df,
                x='date',
                y='value',
                labels={
                    'value': f'{df["name"].iloc[0]} Value',
                    'date': 'Date'
                },
                title=f"{df['name'].iloc[0]} Over Time"
            )
            
            # Add hover template for better data display
            fig.update_traces(
                hovertemplate="<br>".join([
                    "Date: %{x}",
                    "Value: %{y:,.2f}",
                    "<extra></extra>"
                ])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the raw data
            st.subheader("Raw Data")
            st.dataframe(df)
    else:
        st.warning("No economic indicator data available for the selected period.")

def set_price_target(api_key: str):
    with st.expander("Price Target", expanded=False):
        st.header('Price Target')
        # Create or load price targets file
        price_targets_file = 'price_targets.csv'
        
        # Initialize DataFrame
        if os.path.exists(price_targets_file):
            price_targets_df = pd.read_csv(price_targets_file)
            price_targets_df.set_index('symbol', inplace=True)
        else:
            price_targets_df = pd.DataFrame(columns=['symbol', 'price_target', 'creation_date'])
            price_targets_df.set_index('symbol', inplace=True)

        # Input container
        with st.container():
            col1, col2 = st.columns(2)
            
            # Get symbol input
            symbol = col1.text_input("Symbol").upper()
            
            # Get price target input 
            price_target = col2.number_input("Price Target", 
                                              min_value=0.0,
                                              max_value=1000000.0,
                                              value=0.0,
                                              step=0.01)

            col1, col2 = st.columns(2)
            # Add/Update button
            if col1.button("Add/Update Price Target"):
                if symbol:
                    # Update DataFrame
                    price_targets_df.loc[symbol] = [price_target, 
                                                     datetime.now().strftime('%Y-%m-%d')]
                    
                    # Save to CSV
                    price_targets_df.reset_index().to_csv(price_targets_file, index=False)
                    
                    st.success(f"Price target for {symbol} updated successfully!")

            # Remove button
            if col2.button("Remove Symbol"):
                if symbol and symbol in price_targets_df.index:
                    # Remove the symbol
                    price_targets_df.drop(symbol, inplace=True)
                    
                    # Save to CSV
                    price_targets_df.reset_index().to_csv(price_targets_file, index=False)
                    
                    st.success(f"{symbol} removed successfully!")
                elif symbol:
                    st.error(f"{symbol} not found in price targets")

    # Display current price targets
    if not price_targets_df.empty:
        with st.expander("Current Price Targets", expanded=False):
            st.subheader("Current Price Targets")
            
            # Format for display
            display_df = price_targets_df.copy()
            display_df['price_target'] = display_df['price_target'].astype(float).round(2)
            
            st.dataframe(display_df)

def calculate_beta_for_symbol(symbol: str, market_benchmark: str, from_date: str, to_date: str, api_key: str, risk_free_rate: float, market_return: float, time_period: str = 'Monthly') -> tuple[float, float, pd.DataFrame]:
    """
    Calculates the beta of a stock against a market benchmark.
    
    Args:
        symbol (str): The stock symbol to calculate beta for.
        market_benchmark (str): The market benchmark symbol.
        from_date (str): The start date for the calculation.
        to_date (str): The end date for the calculation.
        api_key (str): The API key for financial data.
        risk_free_rate (float): The risk-free rate for CAPM calculation.
        market_return (float): The expected market return for CAPM calculation.
        time_period (str, optional): The time period for resampling ('Daily', 'Monthly', 'Quarterly', 'Yearly'). Defaults to 'Monthly'.

    Returns:
        tuple[float, float, pd.DataFrame]: A tuple containing:
            - float: The calculated beta value.
            - float: The expected return calculated using CAPM.
            - pd.DataFrame: The historical price data used for the calculation.
    """
    historical_price = get_prices_from_symbols([symbol, market_benchmark], api_key, 'close', from_date=from_date, to_date=to_date)
    
    resampled_price = historical_price.copy()
    if time_period == 'Daily':
        resampled_price = resampled_price.resample('D').last()
    elif time_period == 'Monthly':
        resampled_price = resampled_price.resample('ME').last()
    elif time_period == 'Quarterly':
        resampled_price = resampled_price.resample('QE').last()
    elif time_period == 'Yearly':
        resampled_price = resampled_price.resample('YE').last()

    returns = resampled_price.pct_change().dropna()
    
    if symbol not in returns.columns or market_benchmark not in returns.columns:
        return 0.0, 0.0, pd.DataFrame()

    beta = calculate_beta(returns[symbol], returns[market_benchmark])
    expected_return = calculate_capm_expected_return(risk_free_rate, beta, market_return)

    return beta, expected_return, historical_price

def plot_metric_percentage_change(plot_data, x_axis, selected_metrics):
    """
    Display a bar chart of the percentage change for each year or quarter for the selected metrics.
    Positive values are green, negative values are red.
    Handles data sorted from largest to smallest (reverse order).
    """
    # Sort the data in ascending order for correct chronological pct_change
    pct_change_df = plot_data[[x_axis] + selected_metrics].copy()
    pct_change_df = pct_change_df.sort_values(by=x_axis, ascending=False)
    pct_change_df.set_index(x_axis, inplace=True)
    pct_change_df = pct_change_df.pct_change() * -100
    pct_change_df = pct_change_df.dropna()
    pct_change_df = pct_change_df.reset_index()

    # Create a bar chart for each metric
    fig = go.Figure()
    for metric in selected_metrics:
        colors = ['green' if v >= 0 else 'red' for v in pct_change_df[metric]]
        fig.add_trace(go.Bar(
            x=pct_change_df[x_axis],
            y=pct_change_df[metric],
            name=metric,
            marker_color=colors,
            text=[f'{v:.2f}%' for v in pct_change_df[metric]],
            textposition='auto',
        ))
    fig.update_layout(
        barmode='group',
        title='Year-over-Year (or Quarter-over-Quarter) Percentage Change',
        xaxis_title=x_axis,
        yaxis_title='Percentage Change (%)',
        legend_title='Metric',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def display_sector_pe_chart(data):
    """
    Display historical sector PE ratio data in a line chart.
    
    Args:
        data (dict or list): Historical sector PE data from the API
    """
    if isinstance(data, dict) and 'data' in data:
        df = data['data']
    else:
        df = data

    if isinstance(df, list):
        import pandas as pd
        df = pd.DataFrame(df)

    if not df.empty:
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Create the line chart
            fig = px.line(
                df,
                x='date',
                y='pe',
                labels={
                    'pe': 'P/E Ratio',
                    'date': 'Date'
                },
                title=f"{df['sector'].iloc[0]} Sector P/E Ratio History ({df['exchange'].iloc[0]})"
            )
            
            # Add hover template for better data display
            fig.update_traces(
                hovertemplate="<br>".join([
                    "Date: %{x}",
                    "P/E Ratio: %{y:.2f}",
                    "<extra></extra>"
                ])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the raw data
            st.subheader("Raw Data")
            st.dataframe(df)
    else:
        st.warning("No sector PE ratio data available for the selected period.")

def display_industry_pe_chart(data):
    """
    Display historical industry PE ratio data in a line chart.
    
    Args:
        data (dict or list): Historical industry PE data from the API
    """
    if isinstance(data, dict) and 'data' in data:
        df = data['data']
    else:
        df = data

    if isinstance(df, list):
        import pandas as pd
        df = pd.DataFrame(df)

    if not df.empty:
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Create the line chart
            fig = px.line(
                df,
                x='date',
                y='pe',
                labels={
                    'pe': 'P/E Ratio',
                    'date': 'Date'
                },
                title=f"{df['industry'].iloc[0]} Industry P/E Ratio History ({df['exchange'].iloc[0]})"
            )
            
            # Add hover template for better data display
            fig.update_traces(
                hovertemplate="<br>".join([
                    "Date: %{x}",
                    "P/E Ratio: %{y:.2f}",
                    "<extra></extra>"
                ])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the raw data
            st.subheader("Raw Data")
            st.dataframe(df)
    else:
        st.warning("No industry PE ratio data available for the selected period.")

def create_bar_chart(data, title):
    df = data.loc[0].reset_index()
    df.columns = ['Metric', 'Value']
    df = df[df['Metric'] != 'symbol']  # Drop 'symbol' column
    available_metrics = df['Metric'].tolist()
    chart_cols = st.multiselect(
        f"Select metrics for {title} Chart",
        available_metrics,
        default=available_metrics[:3],
        format_func=lambda x: x.split('_')[-1] if '_' in x else x
    )
    if chart_cols:
        plot_df = df[df['Metric'].isin(chart_cols)]
        fig = px.bar(
            plot_df,
            x='Metric',
            y='Value',
            title=title,
            labels={'Value': 'Amount', 'Metric': 'Metric'},
            color='Metric',  # Assign different color to each bar
            barmode='overlay',  # Overlay bars to make them touch
            width=10  # Set bar width to a valid value
        )
        st.plotly_chart(fig)
