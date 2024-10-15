import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import product
import pandas as pd
import warnings
import multiprocessing as mp
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def run_optimization(params, x_start=30, C=200, C_B=50, initial_prices_B=300.0, initial_prices_E=150.0):
    """
    Runs the optimization for a given set of parameters.

    Parameters:
        params (dict): Dictionary containing all necessary parameters.
        x_start (int): Number of days before departure when ticket sales start.
        C (int): Total capacity.
        C_B (int): Capacity allocated to Business class.
        initial_prices_B (float): Initial Business class price.
        initial_prices_E (float): Initial Economy class price.

    Returns:
        dict: Results containing total revenue, ticket sales, and optimized prices.
    """
    # Unpack parameters
    g_B = params['g_B']
    d_B = params['d_B']
    h_B = params['h_B']
    a_B = params['a_B']
    b_B = params['b_B']
    
    g_E = params['g_E']
    d_E = params['d_E']
    h_E = params['h_E']
    a_E = params['a_E']
    b_E = params['b_E']
    
    # Derived parameters
    C_E = C - C_B
    days = np.arange(x_start, -1, -1)  # Days before departure: 30, 29, ..., 0
    
    # Define demand functions
    def demand_B(x):
        return (g_B * x + d_B) * np.exp(-h_B * x)
    
    def demand_E(x):
        return (g_E * x + d_E) * np.exp(-h_E * x)
    
    # Define probability functions
    def probability_B(y, x):
        return np.exp(-y * (a_B + b_B * x))
    
    def probability_E(y, x):
        return np.exp(-y * (a_E + b_E * x))
    
    # Initial price guesses
    y_B_initial = np.full_like(days, initial_prices_B)
    y_E_initial = np.full_like(days, initial_prices_E)
    
    # Concatenate initial prices for optimization
    initial_prices = np.concatenate([y_B_initial, y_E_initial])
    
    # Objective function: Negative total revenue
    def objective(prices):
        y_B = prices[:len(days)]
        y_E = prices[len(days):]
        
        # Vectorized computation for efficiency
        revenue_B = np.sum(y_B * demand_B(days) * probability_B(y_B, days))
        revenue_E = np.sum(y_E * demand_E(days) * probability_E(y_E, days))
        revenue = revenue_B + revenue_E
        
        return -revenue  # Negative for minimization
    
    # Constraints: Expected sales do not exceed capacities
    def constraint_B(prices):
        y_B = prices[:len(days)]
        expected_sales_B = np.sum(demand_B(days) * probability_B(y_B, days))
        return C_B - expected_sales_B  # Should be >= 0
    
    def constraint_E(prices):
        y_E = prices[len(days):]
        expected_sales_E = np.sum(demand_E(days) * probability_E(y_E, days))
        return C_E - expected_sales_E  # Should be >= 0
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_B},
        {'type': 'ineq', 'fun': constraint_E}
    ]
    
    # Define price bounds: Business ($150-$800), Economy ($50-$400)
    bounds_B = [(150, 800) for _ in days]
    bounds_E = [(50, 400) for _ in days]
    bounds = bounds_B + bounds_E
    
    # Run optimization
    try:
        result = minimize(
            objective,
            initial_prices,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False, 'maxiter': 1000}
        )
        
        if result.success:
            optimized_prices = result.x
            y_B_opt = optimized_prices[:len(days)]
            y_E_opt = optimized_prices[len(days):]
            
            # Calculate expected sales
            expected_sales_B = demand_B(days) * probability_B(y_B_opt, days)
            expected_sales_E = demand_E(days) * probability_E(y_E_opt, days)
            
            total_sales_B = np.sum(expected_sales_B)
            total_sales_E = np.sum(expected_sales_E)
            total_revenue = np.sum(y_B_opt * expected_sales_B) + np.sum(y_E_opt * expected_sales_E)
            
            return {
                'total_revenue': total_revenue,
                'total_sales_B': total_sales_B,
                'total_sales_E': total_sales_E,
                'y_B_opt': y_B_opt,
                'y_E_opt': y_E_opt
            }
        else:
            # Return NaNs if optimization failed
            return {
                'total_revenue': np.nan,
                'total_sales_B': np.nan,
                'total_sales_E': np.nan,
                'y_B_opt': np.full_like(days, np.nan),
                'y_E_opt': np.full_like(days, np.nan)
            }
    except Exception as e:
        # Handle unexpected errors
        return {
            'total_revenue': np.nan,
            'total_sales_B': np.nan,
            'total_sales_E': np.nan,
            'y_B_opt': np.full_like(days, np.nan),
            'y_E_opt': np.full_like(days, np.nan)
        }

def perform_grid_search(params_list, x_start=30, C=200, C_B=50):
    """
    Performs a grid search over the provided list of parameter dictionaries using multiprocessing.

    Parameters:
        params_list (list): List of parameter dictionaries.
        x_start (int): Number of days before departure when ticket sales start.
        C (int): Total capacity.
        C_B (int): Capacity allocated to Business class.

    Returns:
        pd.DataFrame: DataFrame containing results for each parameter combination.
    """
    # Define the number of processes (use the number of CPU cores available)
    num_processes = mp.cpu_count()
    
    print(f"Starting grid search with {num_processes} parallel processes over {len(params_list)} parameter combinations...")
    
    # Create a pool of worker processes
    with mp.Pool(processes=num_processes) as pool:
        # Prepare arguments for run_optimization
        args = [(params, x_start, C, C_B, 300.0, 150.0) for params in params_list]
        
        # Map the run_optimization function to the arguments
        results = pool.starmap(run_optimization, args)
    
    print("Grid search completed.")
    
    # Combine results with parameter configurations
    results_df = pd.DataFrame(params_list)
    results_df['total_revenue'] = [res['total_revenue'] for res in results]
    results_df['total_sales_B'] = [res['total_sales_B'] for res in results]
    results_df['total_sales_E'] = [res['total_sales_E'] for res in results]
    
    return results_df

def visualize_grid_search_results(df):
    """
    Visualizes the grid search results.

    Parameters:
        df (pd.DataFrame): DataFrame containing grid search results.
    """
    sns.set(style="whitegrid")
    
    # Scatter plot of total revenue vs. g_B and g_E
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['g_B'], df['g_E'], c=df['total_revenue'], cmap='viridis', marker='o')
    plt.colorbar(scatter, label='Total Revenue')
    plt.xlabel('g_B (Business Demand Slope)')
    plt.ylabel('g_E (Economy Demand Slope)')
    plt.title('Total Revenue across g_B and g_E')
    plt.show()
    
    # Scatter plot of total revenue vs. h_B and h_E
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['h_B'], df['h_E'], c=df['total_revenue'], cmap='plasma', marker='o')
    plt.colorbar(scatter, label='Total Revenue')
    plt.xlabel('h_B (Business Demand Decay)')
    plt.ylabel('h_E (Economy Demand Decay)')
    plt.title('Total Revenue across h_B and h_E')
    plt.show()
    
    # Heatmap for a pair of parameters, e.g., a_B vs a_E
    plt.figure(figsize=(12, 8))
    pivot = df.pivot_table(values='total_revenue', index='a_B', columns='a_E', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap='YlGnBu')
    plt.xlabel('a_E (Economy Probability Slope)')
    plt.ylabel('a_B (Business Probability Slope)')
    plt.title('Average Total Revenue across a_B and a_E')
    plt.show()
    
    # Pairplot colored by total_revenue
    sns.pairplot(df, vars=['g_B', 'd_B', 'h_B', 'g_E', 'd_E', 'h_E'], 
                 hue='total_revenue', palette='viridis', diag_kind='kde', plot_kws={'alpha':0.6})
    plt.show()

def main():
    # Define parameter ranges for grid search (streamlined)
    parameter_grid = {
        # Business Demand Parameters
        'g_B': [2.0],          # Demand slope for Business
        'd_B': [50],           # Demand intercept for Business
        'h_B': [0.05],         # Demand decay for Business
        
        # Business Probability Parameters
        'a_B': [0.1],          # Probability slope for Business
        'b_B': [0.005],        # Probability decay for Business
        
        # Economy Demand Parameters
        'g_E': [5.0],          # Demand slope for Economy
        'd_E': [150],          # Demand intercept for Economy
        'h_E': [0.03],         # Demand decay for Economy
        
        # Economy Probability Parameters
        'a_E': [0.05],         # Probability slope for Economy
        'b_E': [0.002]         # Probability decay for Economy
    }
    
    # To introduce variability, select a few variations for some parameters
    # For example, vary g_B and g_E
    # Adjust the parameter grid accordingly
    # Here, we'll allow g_B and g_E to vary, keeping others constant for simplicity
    parameter_grid = {
        'g_B': [1.8, 2.0, 2.2],          # Demand slope for Business
        'd_B': [50, 60, 70],                      # Demand intercept for Business
        'h_B': [-0.1, -0.05, -0.01],                    # Demand decay for Business
        
        'a_B': [0.1, 0.2, 0.3],                      # Probability slope for Business
        'b_B': [0.005, 0.01, 0.015],                    # Probability decay for Business
        
        'g_E': [4.8, 5.0, 5.2],          # Demand slope for Economy
        'd_E': [150, 160, 170],                      # Demand intercept for Economy
        'h_E': [-0.1, -0.05, -0.01],                     # Demand decay for Economy
        
        'a_E': [0.05, 0.1, 0.15],                     # Probability slope for Economy
        'b_E': [0.002, 0.004, 0.006]                     # Probability decay for Economy
    }
    
    # Generate all combinations using itertools.product
    keys, values = zip(*parameter_grid.items())
    params_list = [dict(zip(keys, v)) for v in product(*values)]
    
    # Perform grid search with multiprocessing
    results_df = perform_grid_search(params_list)
    
    # Save results to CSV for further analysis if needed
    results_df.to_csv('grid_search_results.csv', index=False)
    
    # Display summary statistics
    print("\nSummary of Grid Search Results:")
    print(results_df[['total_revenue', 'total_sales_B', 'total_sales_E']].describe())
    
    # Identify the best parameter set
    best_result = results_df.loc[results_df['total_revenue'].idxmax()]
    print("\nBest Parameter Set:")
    print(best_result)
    
    # Visualize the results
    visualize_grid_search_results(results_df)
    
    # Optional: Plot optimized prices for the best parameter set
    # Re-run optimization to get optimized prices if not already stored
    best_params = best_result.to_dict()
    # Remove the total_revenue and sales entries
    for key in ['total_revenue', 'total_sales_B', 'total_sales_E']:
        best_params.pop(key, None)
    
    # Get optimized prices from results_df
    # Since we stored only summary statistics, re-run optimization to get prices
    best_optimization = run_optimization(best_params)
    
    if not np.isnan(best_optimization['total_revenue']):
        days = np.arange(30, -1, -1)
        y_B_opt = best_optimization['y_B_opt']
        y_E_opt = best_optimization['y_E_opt']
        
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(days, y_B_opt, label='Business Class Price', color='blue')
        plt.xlabel('Days Before Departure')
        plt.ylabel('Price ($)')
        plt.title('Optimized Business Class Prices (Best Parameters)')
        plt.gca().invert_xaxis()
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(days, y_E_opt, label='Economy Class Price', color='green')
        plt.xlabel('Days Before Departure')
        plt.ylabel('Price ($)')
        plt.title('Optimized Economy Class Prices (Best Parameters)')
        plt.gca().invert_xaxis()
        plt.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
