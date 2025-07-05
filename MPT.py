import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
import plotly.express as px

def portfolio_stats_from_data(weights: pd.Series, returns_data: pd.DataFrame, risk_free_rate: float = 0.042):
    """
    Compute expected portfolio return and variance from historical returns.

    Parameters:
    - weights: 1D numpy array of portfolio weights (shape: n,)
    - returns_data: 2D numpy array of historical returns (shape: t x n),
                    where t = number of time periods, n = number of assets.

    Returns:
    - expected_return: float
    - portfolio_variance: float
    """
    # Make sure weights is a 1D array
    weights = weights.values

    # Calculate mean returns for each asset
    mean_returns = returns_data.mean(axis=0)

    # Calculate covariance matrix of asset returns
    cov_matrix = returns_data.cov()

    # Compute expected return and variance
    expected_return = np.dot(weights, mean_returns) * 252
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights)) * 252

    sharpe_ratio = (expected_return - risk_free_rate) / np.sqrt(portfolio_variance)

    return expected_return, portfolio_variance, sharpe_ratio

def compute_portfolio_beta(weights, returns_data, market_returns):
    portfolio_returns = returns_data @ weights
    cov_with_market = np.cov(portfolio_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    beta = cov_with_market / market_variance
    return beta

def calculate_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.042, negative=False):
    port_return = np.dot(weights, expected_returns)
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    if negative:
        return -(port_return - risk_free_rate) / port_std
    else:
        return (port_return - risk_free_rate) / port_std

def maximize_sharpe(expected_returns, cov_matrix, risk_free_rate=0.042):
    n = len(expected_returns)

    initial_weights = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    result = minimize(calculate_sharpe_ratio, initial_weights, method='SLSQP',
                      args=(expected_returns, cov_matrix, risk_free_rate, True),
                      bounds=bounds, constraints=constraints)
    return result.x, -result.fun

def calculate_sortino_ratio(weights, expected_returns, returns_data, risk_free_rate=0.042, negative=False):
    """
    Calculate the Sortino ratio for a portfolio.
    
    Args:
        weights: Portfolio weights
        expected_returns: Expected returns for each asset
        returns_data: Historical returns data
        risk_free_rate: Risk-free rate
        negative: Whether to return negative ratio (for minimization)
    
    Returns:
        float: Sortino ratio
    """
    port_return = np.dot(weights, expected_returns)
    # Calculate downside returns (returns below risk-free rate)
    portfolio_returns = returns_data @ weights
    downside_returns = portfolio_returns[portfolio_returns < risk_free_rate]
    if len(downside_returns) == 0:
        return 0 if negative else float('inf')
    downside_std = np.sqrt(np.mean(downside_returns**2))
    if downside_std == 0:
        return 0 if negative else float('inf')
    sortino = (port_return - risk_free_rate) / downside_std
    return -sortino if negative else sortino

def maximize_sortino(expected_returns, returns_data, risk_free_rate=0.042):
    """
    Maximize the Sortino ratio of a portfolio.
    """
    n = len(expected_returns)
    initial_weights = np.ones(n) / n
    bounds = [(0, 1)] * n
    # Use a constraint that returns a float, not a boolean, to avoid numpy boolean subtract error
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    result = minimize(calculate_sortino_ratio, initial_weights, method='SLSQP',
                     args=(expected_returns, returns_data, risk_free_rate, True),
                     bounds=bounds, constraints=constraints)
    return result.x, -result.fun

def calculate_treynor_ratio(weights, expected_returns, returns_data, market_returns, risk_free_rate=0.042, negative=False):
    """
    Calculate the Treynor ratio for a portfolio.
    
    Args:
        weights: Portfolio weights
        expected_returns: Expected returns for each asset
        returns_data: Historical returns data
        market_returns: Market returns
        risk_free_rate: Risk-free rate
        negative: Whether to return negative ratio (for minimization)
    
    Returns:
        float: Treynor ratio
    """
    port_return = np.dot(weights, expected_returns)
    portfolio_returns = returns_data @ weights
    beta = np.cov(portfolio_returns, market_returns)[0, 1] / np.var(market_returns)
    if beta == 0:
        return 0 if negative else float('inf')
    treynor = (port_return - risk_free_rate) / beta
    return -treynor if negative else treynor

def maximize_treynor(expected_returns, returns_data, market_returns, risk_free_rate=0.042):
    """
    Maximize the Treynor ratio of a portfolio.
    """
    n = len(expected_returns)
    initial_weights = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    result = minimize(calculate_treynor_ratio, initial_weights, method='SLSQP',
                     args=(expected_returns, returns_data, market_returns, risk_free_rate, True),
                     bounds=bounds, constraints=constraints)
    return result.x, -result.fun

def calculate_calmar_ratio(weights, expected_returns, returns_data, risk_free_rate=0.042, negative=False):
    """
    Calculate the Calmar ratio for a portfolio.
    
    Args:
        weights: Portfolio weights
        expected_returns: Expected returns for each asset
        returns_data: Historical returns data
        risk_free_rate: Risk-free rate
        negative: Whether to return negative ratio (for minimization)
    
    Returns:
        float: Calmar ratio
    """
    port_return = np.dot(weights, expected_returns)
    portfolio_returns = returns_data @ weights
    # Calculate maximum drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = np.min(drawdowns)
    if max_drawdown == 0:
        return 0 if negative else float('inf')
    calmar = (port_return - risk_free_rate) / abs(max_drawdown)
    return -calmar if negative else calmar

def maximize_calmar(expected_returns, returns_data, risk_free_rate=0.042):
    """
    Maximize the Calmar ratio of a portfolio.
    """
    n = len(expected_returns)
    initial_weights = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    result = minimize(calculate_calmar_ratio, initial_weights, method='SLSQP',
                     args=(expected_returns, returns_data, risk_free_rate, True),
                     bounds=bounds, constraints=constraints)
    return result.x, -result.fun

def calculate_omega_ratio(weights, returns_data, risk_free_rate=0.042, negative=False):
    """
    Calculate the Omega ratio for a portfolio.
    
    Args:
        weights: Portfolio weights
        expected_returns: Expected returns for each asset
        returns_data: Historical returns data
        risk_free_rate: Risk-free rate
        negative: Whether to return negative ratio (for minimization)
    
    Returns:
        float: Omega ratio
    """
    portfolio_returns = returns_data @ weights
    excess_returns = portfolio_returns - risk_free_rate
    positive_returns = np.sum(excess_returns[excess_returns > 0])
    negative_returns = abs(np.sum(excess_returns[excess_returns < 0]))
    if negative_returns == 0:
        return 0 if negative else float('inf')
    omega = positive_returns / negative_returns
    return -omega if negative else omega

def maximize_omega(expected_returns, returns_data, risk_free_rate=0.042):
    """
    Maximize the Omega ratio of a portfolio.
    """
    n = len(expected_returns)
    initial_weights = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    result = minimize(calculate_omega_ratio, initial_weights, method='SLSQP',
                     args=(expected_returns, returns_data, risk_free_rate, True),
                     bounds=bounds, constraints=constraints)
    return result.x, -result.fun

def calculate_information_ratio(weights, returns_data, benchmark_returns, negative=False):
    """
    Calculate the Information ratio for a portfolio.
    
    Args:
        weights: Portfolio weights
        expected_returns: Expected returns for each asset
        returns_data: Historical returns data
        benchmark_returns: Benchmark returns
        negative: Whether to return negative ratio (for minimization)
    
    Returns:
        float: Information ratio
    """
    portfolio_returns = returns_data @ weights
    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = np.std(excess_returns)
    if tracking_error == 0:
        return 0 if negative else float('inf')
    information_ratio = np.mean(excess_returns) / tracking_error
    return -information_ratio if negative else information_ratio

def maximize_information_ratio(expected_returns, returns_data, benchmark_returns):
    """
    Maximize the Information ratio of a portfolio.
    """
    n = len(expected_returns)
    initial_weights = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    result = minimize(calculate_information_ratio, initial_weights, method='SLSQP',
                     args=(expected_returns, returns_data, benchmark_returns, True),
                     bounds=bounds, constraints=constraints)
    return result.x, -result.fun

def calculate_portfolio_performance(weights, expected_returns, cov_matrix):
    """
    Calculate the expected return and standard deviation of a portfolio.
    """
    port_return = np.dot(weights, expected_returns)
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_std

def generate_random_portfolios(num_portfolios, expected_returns, cov_matrix, risk_free_rate=0.042):
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(expected_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        port_return, port_std = calculate_portfolio_performance(weights, expected_returns, cov_matrix)
        
        results[0,i] = port_return
        results[1,i] = port_std
        results[2,i] = (port_return - risk_free_rate) / port_std  # Sharpe Ratio

    return results, weights_record

def plot_efficient_frontier(expected_returns, cov_matrix, num_portfolios=10000, risk_free_rate=0.042):
    results, weights = generate_random_portfolios(num_portfolios, expected_returns, cov_matrix, risk_free_rate)

    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        'Risk': results[1, :],
        'Return': results[0, :],
        'Sharpe Ratio': results[2, :]
    })

    # Plot using Plotly Express
    fig = px.scatter(
        df,
        x='Risk',
        y='Return',
        color='Sharpe Ratio',
        color_continuous_scale='YlGnBu',
        title='Efficient Frontier',
        labels={'Risk': 'Risk (Standard Deviation)', 'Return': 'Return'}
    )

    fig.show()

