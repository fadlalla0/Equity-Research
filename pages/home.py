import streamlit as st
from utils import *
from concurrent.futures import ThreadPoolExecutor
import plotly.express as px
from datetime import datetime, timedelta

# st.set_page_config(layout='wide')

st.title("🏠 Home")

# CSV File Upload Section
with st.sidebar:
    st.header('📁 Portfolio Upload')

    uploaded_file = st.file_uploader(
        "Upload your portfolio CSV file",
        type=['csv'],
        help="Upload a CSV file with columns: symbol (required), shares (optional, float values)"
    )
    portfolio_data = None
    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file
            portfolio_data = pd.read_csv(uploaded_file)
            
            # Validate required columns
            if 'symbol' not in portfolio_data.columns:
                st.error("CSV file must contain a 'symbol' column")
            else:
                # Check if shares column exists, if not create it with default value 1
                if 'shares' not in portfolio_data.columns:
                    portfolio_data['shares'] = 1.0
                    st.info("No 'shares' column found. Using default value of 1.0 for all stocks.")
                else:
                    # Validate that shares column contains numeric values
                    try:
                        portfolio_data['shares'] = pd.to_numeric(portfolio_data['shares'], errors='coerce')
                        if portfolio_data['shares'].isna().any():
                            st.warning("Some share values could not be converted to numbers. Using 1.0 for those entries.")
                            portfolio_data['shares'] = portfolio_data['shares'].fillna(1.0)
                    except Exception as e:
                        st.error("Shares column must contain numeric values")
                        portfolio_data = None
                
                st.success(f"Successfully uploaded portfolio with {len(portfolio_data)} stocks")
                
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.info("Please ensure your CSV file has the correct format with columns: symbol (required) and shares (optional, float values)")

st.divider()

if portfolio_data is not None:
    api_key = st.session_state.get('api_key')
    if not api_key:
        st.warning("Please enter your API key in the sidebar to load data.")
    else:
        # Run data fetching in parallel
        with ThreadPoolExecutor() as executor:

            # Set from_date to one day before now
            from_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')

            treasury_future = executor.submit(treasury_yield, from_date, to_date, api_key)
            main_data_future = executor.submit(load_main_data, portfolio_data, api_key)

            # Get the result of main_data_future to use in subsequent calls
            data, info = main_data_future.result()

            key_data_future = executor.submit(get_key_data, data, api_key)
            financial_scores_future = executor.submit(get_financial_scores_from_symbols, list(data.index), api_key)

            # Get results
            treasury_data = treasury_future.result()[0]
            key_metrics, ratios = key_data_future.result()
            financial_scores = financial_scores_future.result()

        # Display Treasury Yields
        cols = st.columns(6)
        maturities = ['1Y', '2Y', '3Y', '5Y','10Y', '30Y']
        rates = [
            treasury_data['year1'], treasury_data['year2'], treasury_data['year3'],
            treasury_data['year5'], treasury_data['year10'], treasury_data['year30']
        ]

        for col, maturity, rate in zip(cols, maturities, rates):
            with col:
                st.metric(maturity, f"{rate:.2f}%")

        st.divider()

        basic_metrics(data)
        st.divider()

        st.header('Financial Scores')
        st.dataframe(financial_scores)

        st.divider()

        key_metrics = key_metrics.set_index('symbol')
        ratios = ratios.set_index('symbol')

        st.header('Key Metrics')
        selected_key_metrics = st.multiselect('Select the key metrics', key_metrics.columns, default=key_metrics.columns[:5])
        st.dataframe(key_metrics[selected_key_metrics])

        st.divider()

        st.header('Ratios')
        selected_ratios = st.multiselect('Select the ratios', ratios.columns, default=ratios.columns[:5])
        st.dataframe(ratios[selected_ratios])

        st.divider()

        display_compounded_returns_chart(api_key)
