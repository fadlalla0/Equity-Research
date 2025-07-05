# Equity Research Dashboard

An interactive web application built with Streamlit for performing equity research and financial analysis. This dashboard provides tools for stock analysis, portfolio management, and economic indicator tracking.

## Features

-   **API Key Authentication**: Securely enter and use your Financial Modeling Prep (FMP) API key.
-   **Home Page**: Upload your portfolio via CSV and view a high-level overview, including key metrics, financial scores, and a compounded returns chart comparing your assets to major ETFs.
-   **Stock Search**: Search for individual stocks to get detailed information, including:
    -   Historical price charts with financial ratios.
    -   Financial scores and statements (Income, Balance Sheet, Cash Flow).
    -   An interactive Peter Lynch-style stock checklist.
-   **Portfolio Management**:
    -   Calculate a stock's Beta against a market benchmark.
    -   Determine expected returns using the Capital Asset Pricing Model (CAPM).
    -   Set and manage price targets for stocks.
-   **Comparison**: Perform side-by-side comparisons of multiple stocks across various financial metrics and statements.
-   **Economics**:
    -   Track historical US Treasury yields.
    -   View major economic indicators (e.g., GDP, CPI, Unemployment Rate).
    -   Analyze historical P/E ratios for different market sectors and industries.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Equity-Research
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Key:**
    -   This application requires an API key from [Financial Modeling Prep](https://site.financialmodelingprep.com/developer/docs).
    -   Create a file named `.env` in the root directory of the project.
    -   Add your API key to the `.env` file as follows:
        ```
        FMP_API_KEY="YOUR_API_KEY_HERE"
        ```
    -   Alternatively, you can enter the API key directly into the application's sidebar.

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2.  The application will open in your web browser.

3.  If you have not set the `FMP_API_KEY` in the `.env` file, you will be prompted to enter your API key in the sidebar. The application features will be unlocked once a valid key is provided. 