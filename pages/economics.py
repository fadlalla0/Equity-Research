import streamlit as st
from utils import (
    treasury_yield, display_treasury_yield_chart,
    get_economic_indicators, display_economic_indicator_chart,
    get_historical_sector_pe, display_sector_pe_chart,
    get_historical_industry_pe, display_industry_pe_chart,
    get_directory_cached
)
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor

st.title("Economics")

api_key = st.session_state.get('api_key')
if not api_key:
    st.warning("Please enter your API key in the sidebar to view economic data.")
    st.stop()

# Available economic indicators
ECONOMIC_INDICATORS = {
    "GDP": "Gross Domestic Product",
    "realGDP": "Real GDP",
    "nominalPotentialGDP": "Nominal Potential GDP",
    "realGDPPerCapita": "Real GDP Per Capita",
    "federalFunds": "Federal Funds Rate",
    "CPI": "Consumer Price Index",
    "inflationRate": "Inflation Rate",
    "inflation": "Inflation",
    "retailSales": "Retail Sales",
    "consumerSentiment": "Consumer Sentiment",
    "durableGoods": "Durable Goods",
    "unemploymentRate": "Unemployment Rate",
    "totalNonfarmPayroll": "Total Nonfarm Payroll",
    "initialClaims": "Initial Claims",
    "industrialProductionTotalIndex": "Industrial Production Index",
    "newPrivatelyOwnedHousingUnitsStartedTotalUnits": "New Housing Units Started",
    "totalVehicleSales": "Total Vehicle Sales",
    "retailMoneyFunds": "Retail Money Funds",
    "smoothedUSRecessionProbabilities": "US Recession Probabilities",
    "3MonthOr90DayRatesAndYieldsCertificatesOfDeposit": "3-Month CD Rates",
    "commercialBankInterestRateOnCreditCardPlansAllAccounts": "Credit Card Interest Rates",
    "30YearFixedRateMortgageAverage": "30-Year Fixed Mortgage Rate",
    "15YearFixedRateMortgageAverage": "15-Year Fixed Mortgage Rate"
}

# Available treasury rates
TREASURY_RATES = {
    "month1": "1 Month Treasury Rate",
    "month2": "2 Month Treasury Rate",
    "month3": "3 Month Treasury Rate",
    "month6": "6 Month Treasury Rate",
    "year1": "1 Year Treasury Rate",
    "year2": "2 Year Treasury Rate",
    "year3": "3 Year Treasury Rate",
    "year5": "5 Year Treasury Rate",
    "year7": "7 Year Treasury Rate",
    "year10": "10 Year Treasury Rate",
    "year20": "20 Year Treasury Rate",
    "year30": "30 Year Treasury Rate"
}

# Set default date range (last 2 years)
default_to_date = datetime.now()
default_from_date = default_to_date - timedelta(days=730)

# Create tabs for different economic data
tab1, tab2, tab3, tab4 = st.tabs(["Treasury Rates", "Economic Indicators", "Sector PE Ratios", "Industry PE Ratios"])

with tab1:
    st.subheader("US Treasury Rates History")

    # Create columns for layout
    col1, col2 = st.columns(2)

    # Rate selection
    with col1:
        selected_rates = st.multiselect(
            "Select Treasury Rates",
            options=list(TREASURY_RATES.keys()),
            format_func=lambda x: f"{x} - {TREASURY_RATES[x]}",
            default=["year2", "year10"]
        )

    # Date range selection
    with col2:
        from_date_input = st.date_input(
            "From Date",
            value=default_from_date,
            min_value=datetime(1990, 1, 1),
            max_value=default_to_date,
            key="treasury_from_date"
        )
        to_date_input = st.date_input(
            "To Date",
            value=default_to_date,
            min_value=from_date_input,
            max_value=default_to_date,
            key="treasury_to_date"
        )

    # Convert dates to string format
    from_date = from_date_input.strftime('%Y-%m-%d')
    to_date = to_date_input.strftime('%Y-%m-%d')

    # Fetch and display treasury yield data
    if selected_rates:
        data = treasury_yield(from_date, to_date, api_key)
        display_treasury_yield_chart(data, selected_rates)
    else:
        st.info("Please select at least one treasury rate to display.")

with tab2:
    st.subheader("Economic Indicators")

    # Create columns for layout
    col1, col2 = st.columns(2)

    # Indicator selection
    with col1:
        selected_indicator = st.selectbox(
            "Select Economic Indicator",
            options=list(ECONOMIC_INDICATORS.keys()),
            format_func=lambda x: f"{x} - {ECONOMIC_INDICATORS[x]}"
        )

    # Date range selection
    with col2:
        from_date_input = st.date_input(
            "From Date",
            value=default_from_date,
            min_value=datetime(1990, 1, 1),
            max_value=default_to_date,
            key="indicator_from_date"
        )
        to_date_input = st.date_input(
            "To Date",
            value=default_to_date,
            min_value=from_date_input,
            max_value=default_to_date,
            key="indicator_to_date"
        )

    # Convert dates to string format
    from_date = from_date_input.strftime('%Y-%m-%d')
    to_date = to_date_input.strftime('%Y-%m-%d')

    # Fetch and display data
    if selected_indicator:
        data = get_economic_indicators(selected_indicator, from_date, to_date, api_key)
        display_economic_indicator_chart(data)

with tab3:
    st.subheader("Historical Sector PE Ratios")

    # Get available sectors
    sectors_data = get_directory_cached("available-sectors", api_key)

    # Define major US exchanges
    MAJOR_EXCHANGES = ["NASDAQ", "NYSE", "AMEX"]

    # Create columns for layout
    col1, col2, col3 = st.columns(3)

    # Sector selection
    with col1:
        if sectors_data and isinstance(sectors_data, list):
            sectors = [sector['sector'] for sector in sectors_data]
            selected_sectors = st.multiselect(
                "Select Sectors",
                options=sectors,
                default=sectors[:2] if len(sectors) > 1 else sectors,
                key="sector_pe_sectors"
            )
        else:
            st.error("Could not load sectors data")

    # Exchange selection
    with col2:
        selected_exchange = st.selectbox(
            "Select Exchange",
            options=MAJOR_EXCHANGES,
            key="sector_pe_exchange"
        )

    # Date range selection
    with col3:
        from_date_input = st.date_input(
            "From Date",
            value=default_from_date,
            min_value=datetime(1990, 1, 1),
            max_value=default_to_date,
            key="sector_pe_from_date"
        )
        to_date_input = st.date_input(
            "To Date",
            value=default_to_date,
            min_value=from_date_input,
            max_value=default_to_date,
            key="sector_pe_to_date"
        )

    # Convert dates to string format
    from_date = from_date_input.strftime('%Y-%m-%d')
    to_date = to_date_input.strftime('%Y-%m-%d')

    # Fetch and display data
    if selected_sectors and selected_exchange:
        def fetch_sector_data(sector):
            data = get_historical_sector_pe(sector, selected_exchange, from_date, to_date, api_key)
            if data and isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
                df['sector'] = sector
                return df
            return None

        # Fetch data in parallel
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(fetch_sector_data, selected_sectors))
        
        # Filter out None results and combine data
        combined_data = [df for df in results if df is not None]
        
        if combined_data:
            # Combine all data
            combined_df = pd.concat(combined_data)
            # Create the line chart
            fig = px.line(
                combined_df,
                x='date',
                y='pe',
                color='sector',
                labels={
                    'pe': 'P/E Ratio',
                    'date': 'Date',
                    'sector': 'Sector'
                },
                title=f"Sector P/E Ratio History ({selected_exchange})"
            )
            
            # Add hover template for better data display
            fig.update_traces(
                hovertemplate="<br>".join([
                    "Date: %{x}",
                    "P/E Ratio: %{y:.2f}",
                    "Sector: %{fullData.name}",
                    "<extra></extra>"
                ])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the raw data
            st.subheader("Raw Data")
            st.dataframe(combined_df)
        else:
            st.warning("No sector PE ratio data available for the selected period.")
    else:
        st.info("Please select at least one sector and an exchange.")

with tab4:
    st.subheader("Historical Industry PE Ratios")

    # Get available industries
    industries_data = get_directory_cached("available-industries", api_key)

    # Create columns for layout
    col1, col2, col3 = st.columns(3)

    # Industry selection
    with col1:
        if industries_data and isinstance(industries_data, list):
            industries = [industry['industry'] for industry in industries_data]
            selected_industries = st.multiselect(
                "Select Industries",
                options=industries,
                default=industries[:2] if len(industries) > 1 else industries,
                key="industry_pe_industries"
            )
        else:
            st.error("Could not load industries data")

    # Exchange selection
    with col2:
        selected_exchange = st.selectbox(
            "Select Exchange",
            options=MAJOR_EXCHANGES,
            key="industry_pe_exchange"
        )

    # Date range selection
    with col3:
        from_date_input = st.date_input(
            "From Date",
            value=default_from_date,
            min_value=datetime(1990, 1, 1),
            max_value=default_to_date,
            key="industry_pe_from_date"
        )
        to_date_input = st.date_input(
            "To Date",
            value=default_to_date,
            min_value=from_date_input,
            max_value=default_to_date,
            key="industry_pe_to_date"
        )

    # Convert dates to string format
    from_date = from_date_input.strftime('%Y-%m-%d')
    to_date = to_date_input.strftime('%Y-%m-%d')

    # Fetch and display data
    if selected_industries and selected_exchange:
        def fetch_industry_data(industry):
            data = get_historical_industry_pe(industry, selected_exchange, from_date, to_date, api_key)
            if data and isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
                df['industry'] = industry
                return df
            return None

        # Fetch data in parallel
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(fetch_industry_data, selected_industries))
        
        # Filter out None results and combine data
        combined_data = [df for df in results if df is not None]
        
        if combined_data:
            # Combine all data
            combined_df = pd.concat(combined_data)
            # Create the line chart
            fig = px.line(
                combined_df,
                x='date',
                y='pe',
                color='industry',
                labels={
                    'pe': 'P/E Ratio',
                    'date': 'Date',
                    'industry': 'Industry'
                },
                title=f"Industry P/E Ratio History ({selected_exchange})"
            )
            
            # Add hover template for better data display
            fig.update_traces(
                hovertemplate="<br>".join([
                    "Date: %{x}",
                    "P/E Ratio: %{y:.2f}",
                    "Industry: %{fullData.name}",
                    "<extra></extra>"
                ])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the raw data
            st.subheader("Raw Data")
            st.dataframe(combined_df)
        else:
            st.warning("No industry PE ratio data available for the selected period.")
    else:
        st.info("Please select at least one industry and an exchange.")
