import streamlit as st
from utils import *
from concurrent.futures import ThreadPoolExecutor
import plotly.express as px
from datetime import datetime

st.title("⚖️ Comparison")

symbols = st.text_input("Enter Symbols", placeholder="Enter Symbols")
period = st.radio(
    "Choose between annual and quarterly financial statements:",
    options=["Annual 📅", "Quarterly 📆"],
    index=0,
    horizontal=True,
    help="Select the type of financial statement period you want to display."
)
period = "annual" if "Annual" in period else "quarter"

if symbols:
    api_key = st.session_state.get("api_key")
    if not api_key:
        st.warning("Please enter your API key in the sidebar to compare stocks.")
        st.stop()

    symbols_list = symbols.split(",")
    symbols_list = [symbol.strip() for symbol in symbols_list]
    symbols_list = [symbol for symbol in symbols_list if symbol != ""]

    historical_ratios = {}
    balance_sheets = {}
    income_statements = {}
    cash_flows = {}
    margins = {}
    key_metrics_ttm = {}
    key_metrics_annual = {}
    ratios_ttm = {}
    ratios_annual = {}

    balance_sheets_columns = set()
    income_statements_columns = set()
    cash_flows_columns = set()
    margins_columns = set()
    key_metrics_ttm_columns = set()
    key_metrics_annual_columns = set()
    ratios_ttm_columns = set()
    ratios_annual_columns = set()

    with ThreadPoolExecutor() as executor:
        # Submit all tasks in parallel
        futures = {}
        for symbol in symbols_list:
            futures[(symbol, "historical_ratios")] = executor.submit(get_historical_price_ratio, symbol, api_key)
            futures[(symbol, "balance_sheets")] = executor.submit(get_statement, symbol, api_key, "balance-sheet-statement", period)
            futures[(symbol, "income_statements")] = executor.submit(get_statement, symbol, api_key, "income-statement", period)
            futures[(symbol, "cash_flows")] = executor.submit(get_statement, symbol, api_key, "cash-flow-statement", period)
            futures[(symbol, "margins")] = executor.submit(get_margins_from_symbol, symbol, api_key, period)
            futures[(symbol, 'key_metrics_ttm')] = executor.submit(get_key_metrics, symbol, api_key, True)
            futures[(symbol, 'key_metrics_annual')] = executor.submit(get_key_metrics, symbol, api_key, False)
            futures[(symbol, 'ratios_ttm')] = executor.submit(get_ratios, symbol, api_key, True)
            futures[(symbol, 'ratios_annual')] = executor.submit(get_ratios, symbol, api_key, False)
        
        for (symbol, data_type), future in futures.items():
            try:
                result = future.result()
                if data_type == "historical_ratios":
                    historical_ratios[symbol] = result
                elif data_type == "balance_sheets":
                    balance_sheets[symbol] = result
                    if result is not None:
                        balance_sheets_columns.update(result.columns)
                elif data_type == "income_statements":
                    income_statements[symbol] = result
                    if result is not None:
                        income_statements_columns.update(result.columns)
                elif data_type == "cash_flows":
                    cash_flows[symbol] = result
                    if result is not None:
                        cash_flows_columns.update(result.columns)
                elif data_type == "margins":
                    margins[symbol] = result
                    if result is not None:
                        margins_columns.update(result.columns)
                elif data_type == "key_metrics_ttm":
                    key_metrics_ttm[symbol] = result
                    if result is not None:
                        key_metrics_ttm_columns.update(result.columns)
                elif data_type == "key_metrics_annual":
                    key_metrics_annual[symbol] = result
                    if result is not None:
                        key_metrics_annual_columns.update(result.columns)
                elif data_type == "ratios_ttm":
                    ratios_ttm[symbol] = result
                    if result is not None:
                        ratios_ttm_columns.update(result.columns)
                elif data_type == "ratios_annual":
                    ratios_annual[symbol] = result
                    if result is not None:
                        ratios_annual_columns.update(result.columns)
            except Exception as e:
                st.error(f"Failed to fetch {data_type} for {symbol}: {e}")

    st.divider()
    st.subheader("Balacne Sheets")
    balance_columns = st.multiselect(
        "Select Balance Sheet Columns", 
        balance_sheets_columns)
    if balance_columns:
        for column in balance_columns:
            balance_data = {}
            for symbol in symbols_list:
                balance_data[symbol] = balance_sheets[symbol][column].reset_index(drop=True)
            balance_df = pd.DataFrame(balance_data)
            with st.expander(column, expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(balance_df, hide_index=True)
                with col2:
                    st.plotly_chart(px.line(balance_df.sort_index(ascending=False), x=balance_df.index, y=balance_df.columns, title=column))


    st.divider()
    st.subheader("Income Statements")
    income_columns = st.multiselect(
        "Select Income Statement Columns", 
        income_statements_columns)
    if income_columns:
        for column in income_columns:
            income_data = {}
            for symbol in symbols_list:
                income_data[symbol] = income_statements[symbol][column].reset_index(drop=True)
            income_df = pd.DataFrame(income_data)
            with st.expander(column, expanded=False):   
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(income_df, hide_index=True)
                with col2:
                    st.plotly_chart(px.line(income_df.sort_index(ascending=False), x=income_df.index, y=income_df.columns, title=column))

    st.divider()
    st.subheader("Cash Flows")
    cashflow_columns = st.multiselect(
        "Select Cash Flow Columns", 
        cash_flows_columns)
    if cashflow_columns:
        for column in cashflow_columns:
            cashflow_data = {}
            for symbol in symbols_list:
                cashflow_data[symbol] = cash_flows[symbol][column].reset_index(drop=True)
            cashflow_df = pd.DataFrame(cashflow_data)
            with st.expander(column, expanded=False):   
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(cashflow_df, hide_index=True)
                with col2:
                    st.plotly_chart(px.line(cashflow_df.sort_index(ascending=False), x=cashflow_df.index, y=cashflow_df.columns, title=column))
    st.divider()
    st.subheader("Margins")
    margins_columns = st.multiselect(
        "Select Margins Columns", 
        margins_columns)
    if margins_columns:
        for column in margins_columns:
            margins_data = {}
            for symbol in symbols_list:
                margins_data[symbol] = margins[symbol][column].reset_index(drop=True)
            margins_df = pd.DataFrame(margins_data)
            with st.expander(column, expanded=False):   
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(margins_df, hide_index=True)
                with col2:
                    st.plotly_chart(px.line(margins_df.sort_index(ascending=False), x=margins_df.index, y=margins_df.columns, title=column))

    st.divider()
    st.subheader("Key Metrics Annual")
    key_metrics_columns = st.multiselect(
        "Select Key Metrics Columns", 
        key_metrics_annual_columns)
    if key_metrics_annual_columns:
        for column in key_metrics_columns:
            key_metrics_data = {}
            for symbol in symbols_list:
                key_metrics_data[symbol] = key_metrics_annual[symbol][column].reset_index(drop=True)
            key_metrics_df = pd.DataFrame(key_metrics_data)
            with st.expander(column, expanded=False):   
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(key_metrics_df, hide_index=True)
                with col2:
                    st.plotly_chart(px.line(key_metrics_df.sort_index(ascending=False), x=key_metrics_df.index, y=key_metrics_df.columns, title=column))

    st.divider()
    st.subheader("Key Metrics TTM")
    key_metrics_columns = st.multiselect(
        "Select Key Metrics Columns", 
        key_metrics_ttm_columns)
    if key_metrics_ttm_columns:
        for column in key_metrics_columns:
            key_metrics_data = {}
            for symbol in symbols_list:
                key_metrics_data[symbol] = key_metrics_ttm[symbol][column].reset_index(drop=True)
            key_metrics_df = pd.DataFrame(key_metrics_data)
            with st.expander(column, expanded=False):   
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(key_metrics_df, hide_index=True)
                with col2:
                    fig = px.bar(
                        key_metrics_df.sort_index(ascending=False),
                        x=key_metrics_df.index,
                        y=key_metrics_df.columns,
                        title=f"{column} (TTM)",
                        barmode='group'
                    )
                    st.plotly_chart(fig)

    st.divider()
    st.subheader("Ratios Annual")
    ratios_columns = st.multiselect(
        "Select Ratios Columns", 
        ratios_annual_columns)
    if ratios_annual_columns:
        for column in ratios_columns:
            ratios_data = {}
            for symbol in symbols_list:
                ratios_data[symbol] = ratios_annual[symbol][column].reset_index(drop=True)
            ratios_df = pd.DataFrame(ratios_data)
            with st.expander(column, expanded=False):   
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(ratios_df, hide_index=True)
                with col2:
                    st.plotly_chart(px.line(ratios_df.sort_index(ascending=False), x=ratios_df.index, y=ratios_df.columns, title=column))

    st.divider()
    st.subheader("Ratios TTM")
    ratios_columns = st.multiselect(
        "Select Ratios Columns", 
        ratios_ttm_columns)
    if ratios_ttm_columns:
        for column in ratios_columns:
            ratios_data = {}
            for symbol in symbols_list:
                ratios_data[symbol] = ratios_ttm[symbol][column].reset_index(drop=True)
            ratios_df = pd.DataFrame(ratios_data)
            with st.expander(column, expanded=False):   
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(ratios_df, hide_index=True)
                with col2:
                    # For TTM, show a grouped bar chart where each symbol is a separate bar for each period
                    fig = px.bar(
                        ratios_df.sort_index(ascending=False),
                        x=ratios_df.index,
                        y=ratios_df.columns,
                        title=f"{column} (TTM)",
                        barmode='group'
                    )
                    st.plotly_chart(fig)