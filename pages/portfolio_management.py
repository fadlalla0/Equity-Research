import plotly.express as px
import streamlit as st
from utils import *
import pandas as pd
from helper_functions import calculate_capm_expected_return

st.title("📈 Portfolio Management")

# Add input fields for CAPM parameters
st.sidebar.header('CAPM Parameters')
risk_free_rate = st.sidebar.number_input('Risk-Free Rate (%)', min_value=0.0, max_value=100.0, value=3.0) / 100
beta_input = st.sidebar.number_input('Beta', min_value=0.0, value=1.0)
market_return = st.sidebar.number_input('Market Return (%)', min_value=0.0, max_value=100.0, value=8.0) / 100

# Calculate expected return using CAPM
expected_return = calculate_capm_expected_return(risk_free_rate, beta_input, market_return)

# Display the expected return
st.sidebar.write(f'Expected Return: {expected_return:.2%}')

api_key = st.session_state.get('api_key')
if not api_key:
    st.warning("Please enter your API key in the sidebar to use this page.")
else:
    set_price_target(api_key)

    st.divider()

    # Add input fields for symbol and market benchmark
    st.header('Beta Calculation')
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input('Stock Symbol', placeholder='e.g. AAPL')
        from_date = st.date_input('From Date', value=datetime(datetime.now().year - 5, datetime.now().month, datetime.now().day).date(), min_value=datetime(1970, 1, 1), max_value=datetime.today().date())

    with col2:
        market_benchmark = st.text_input('Market Benchmark', value='^GSPC', help='Default is S&P 500 (^GSPC)')
        to_date = st.date_input('To Date', value=datetime.today().date(), min_value=datetime(1970, 1, 1), max_value=datetime.today().date())

    # Add a selectbox for the time period
    time_period = st.selectbox('Period', ['Daily', 'Monthly', 'Quarterly', 'Yearly'], index=1)

    if symbol and market_benchmark:
        beta, expected_return, historical_price = calculate_beta_for_symbol(symbol, market_benchmark, from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d'), api_key, risk_free_rate, market_return, time_period)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f'Beta: {beta:.2f}')
        with col2:
            st.write(f"Average Expected Return (CAPM): {expected_return:.2%}")

        st.dataframe(historical_price)


