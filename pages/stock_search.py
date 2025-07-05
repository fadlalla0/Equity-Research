from datetime import datetime
import streamlit as st
import plotly.express as px
from utils import *
# st.set_page_config(layout='wide')


st.title("🔍 Stock Search")
# Your search code here
symbol = st.text_input(label='Symbol for search', placeholder="Enter a symbol")

col1, col2 = st.columns(2)
from_date = col1.date_input("From Date", value=datetime(datetime.now().year - 5, datetime.now().month, datetime.now().day).date(), min_value=datetime(1970, 1, 1), max_value=datetime.today().date())
to_date = col2.date_input("To Date", value=datetime.today().date(), min_value=datetime(1970, 1, 1), max_value=datetime.today().date())

st.divider()

api_key = st.session_state.get('api_key')

try:
    if symbol != "":
        if not api_key:
            st.warning("Please enter your API key in the sidebar to search for a stock.")
        else:
            # Add update button
            if st.button("Update Historical Data"):
                historical_price = get_historical_price(symbol, api_key, update=True)
                st.success("Historical data updated successfully!")

            # Fetch historical price and financial statements in parallel
            with ThreadPoolExecutor() as executor:
                # Start both tasks concurrently
                historical_price_future = executor.submit(get_historical_price, symbol, api_key)
                historical_price_ratio_future = executor.submit(get_historical_price_ratio, symbol, api_key)
                financial_statements_placeholder = st.empty()
                
                # Get historical price data and display chart
                historical_price = historical_price_future.result()
                filtered_price = historical_price[(historical_price.index >= from_date.strftime("%Y-%m-%d")) & (historical_price.index <= to_date.strftime("%Y-%m-%d"))]

                historical_price_ratio = historical_price_ratio_future.result()
                filtered_price_ratio = historical_price_ratio[(historical_price_ratio.index >= from_date.strftime("%Y-%m-%d")) & (historical_price_ratio.index <= to_date.strftime("%Y-%m-%d"))]
                # Get ratio columns excluding 'close'
                ratio_columns = [col for col in historical_price_ratio.columns if col != 'close']
                selected_ratios = st.multiselect('Select ratios to display', ratio_columns)
                # Create figure with secondary y-axis
                fig = px.line(filtered_price, x=filtered_price.index, y='close', labels={'date': 'Date', 'close': 'Price'})

                # Add selected ratios on secondary y-axis
                for ratio in selected_ratios:
                    fig.add_scatter(x=filtered_price_ratio.index, y=filtered_price_ratio[ratio],
                                  name=ratio, yaxis="y2")

                # Update layout for secondary y-axis
                fig.update_layout(
                    yaxis2=dict(
                        title="Ratio Values",
                        overlaying="y",
                        side="right"
                    )
                )

                st.plotly_chart(fig)
                
                st.divider()
                
                # Display financial scores
                st.subheader("📊 Financial Scores")
                financial_scores = get_financial_scores_from_symbols([symbol], api_key)  # Assuming this function exists in utils.py
                display_financial_scores(financial_scores)

                st.divider()
                # Display financial statements
                display_financial_statements(symbol, api_key)

                st.divider()

                # Stock Checklist
                st.subheader("🔍 Stock Checklist")
                checklist = display_stock_checklist(symbol, api_key)

except Exception as e:
    st.error(f"Failed to load stock data for {symbol}. Error: {e}")