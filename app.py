import streamlit as st
from streamlit_option_menu import option_menu
from utils import *
import plotly.express as px
import os
import requests
from dotenv import load_dotenv

st.set_page_config(layout='wide')

# Load environment variables from .env file
load_dotenv()

# Function to check if API key is valid
def is_api_key_valid(api_key):
    if not api_key:
        return False
    test_url = f"https://financialmodelingprep.com/stable/profile?symbol=AAPL&apikey={api_key}"
    try:
        response = requests.get(test_url)
        if response.status_code == 200:
            data = response.json()
            return bool(data and isinstance(data, list) and 'symbol' in data[0])
        return False
    except Exception:
        return False

def initialize_api_key():
    # Get API key from environment variable
    api_key = os.getenv('FMP_API_KEY', '')
    
    # Create sidebar for API key input
    with st.sidebar:
        st.markdown("## API Configuration")
        input_api_key = st.text_input("Enter FMP API Key:", 
                                     value=api_key if api_key else "",
                                     type="password",
                                     help="Enter your Financial Modeling Prep API key")
        
        # Validate API key when entered
        if input_api_key:
            if is_api_key_valid(input_api_key):
                st.success("API key is valid!")
                # Store valid API key in session state
                st.session_state['api_key'] = input_api_key
            else:
                st.error("Invalid API key. Please check and try again.")
                if 'api_key' in st.session_state:
                    del st.session_state['api_key']

def main():
    # Initialize API key configuration
    initialize_api_key()
    
    # Only show main content if API key is valid
    if 'api_key' in st.session_state:
        pg = st.navigation([
            st.Page("pages/home.py", title="Home", icon="🏠"),
            st.Page("pages/stock_search.py", title="Stock Search", icon="🔍"),
            st.Page("pages/portfolio_management.py", title="Portfolio Management", icon="📈"),
            st.Page("pages/comparison.py", title="Comparison", icon="⚖️"),
            st.Page("pages/economics.py", title="Economics", icon="💰")
        ])
        pg.run()
    else:
        st.warning("Please enter a valid API key in the sidebar to access the application.")

if __name__ == "__main__":
    main()