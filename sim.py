import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
import warnings
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def run_optimization(params, x_start=30, C=200, C_B=50, initial_prices_B=300.0, initial_prices_E=150.0, optimize_classes='both', initial_price_ratio=1.5):
    """
    Runs the optimization for a given set of parameters.

    Parameters:
        params (dict): Dictionary containing all necessary parameters.
        x_start (int): Number of days before departure when ticket sales start.
        C (int): Total capacity.
        C_B (int): Capacity allocated to Business class.
        initial_prices_B (float): Initial Business class price.
        initial_prices_E (float): Initial Economy class price.
        optimize_classes (str): Which classes to optimize ('business', 'economy', 'both').
        initial_price_ratio (float): Initial price ratio between Business and Economy class.

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
    
    # Initialize price vectors based on optimization choice
    if optimize_classes.lower() == 'business':
        y_B_initial = np.full_like(days, initial_prices_B)
        y_E_initial = np.full_like(days, initial_prices_E)  # Prices fixed
        optimize_B = True
        optimize_E = False
    elif optimize_classes.lower() == 'economy':
        y_B_initial = np.full_like(days, initial_prices_B)  # Prices fixed
        y_E_initial = np.full_like(days, initial_prices_E)
        optimize_B = False
        optimize_E = True
    elif optimize_classes.lower() == 'both':
        y_B_initial = np.full_like(days, initial_prices_B)
        y_E_initial = np.full_like(days, initial_prices_E)
        optimize_B = True
        optimize_E = True
    else:
        raise ValueError("Invalid value for 'optimize_classes'. Choose from 'business', 'economy', 'both'.")
    
    # Modify initial prices to include the price ratio
    if optimize_classes.lower() == 'both':
        initial_prices = np.concatenate([y_B_initial, y_E_initial, [initial_price_ratio]])
    elif optimize_classes.lower() == 'business':
        initial_prices = np.concatenate([y_B_initial, [initial_price_ratio]])
    elif optimize_classes.lower() == 'economy':
        initial_prices = np.concatenate([y_E_initial, [initial_price_ratio]])
    
    # Objective function: Negative total revenue
    def objective(prices):
        if optimize_classes.lower() == 'both':
            y_B = prices[:len(days)]
            y_E = prices[len(days):-1]
            price_ratio = prices[-1]
        elif optimize_classes.lower() == 'business':
            y_B = prices[:-1]
            y_E = y_E_initial
            price_ratio = prices[-1]
        elif optimize_classes.lower() == 'economy':
            y_B = y_B_initial
            y_E = prices[:-1]
            price_ratio = prices[-1]

        # Ensure price ratio constraint is met
        y_B = np.maximum(y_B, y_E * price_ratio)

        revenue_B = np.sum(y_B * demand_B(days) * probability_B(y_B, days))
        revenue_E = np.sum(y_E * demand_E(days) * probability_E(y_E, days))
        revenue = revenue_B + revenue_E
        return -revenue  # Negative for minimization
    
    # Constraints
    constraints = []
    
    # Capacity constraints
    if optimize_B:
        def constraint_B(prices):
            if optimize_classes.lower() == 'both':
                y_B = prices[:len(days)]
            else:
                y_B = prices[:-1]
            expected_sales_B = np.sum(demand_B(days) * probability_B(y_B, days))
            return C_B - expected_sales_B  # Should be >= 0
        constraints.append({'type': 'ineq', 'fun': constraint_B})
    
    if optimize_E:
        def constraint_E(prices):
            if optimize_classes.lower() == 'both':
                y_E = prices[len(days):-1]
            elif optimize_classes.lower() == 'economy':
                y_E = prices[:-1]
            else:
                y_E = y_E_initial
            expected_sales_E = np.sum(demand_E(days) * probability_E(y_E, days))
            return C_E - expected_sales_E  # Should be >= 0
        constraints.append({'type': 'ineq', 'fun': constraint_E})
    
    # Price ratio constraint
    def constraint_price_ratio(prices):
        return prices[-1] - 1  # price_ratio should be > 1

    constraints.append({'type': 'ineq', 'fun': constraint_price_ratio})
    
    # Define bounds
    if optimize_classes.lower() == 'both':
        bounds_B = [(120, 800) for _ in days]
        bounds_E = [(50, 400) for _ in days]
        bounds = bounds_B + bounds_E + [(1.01, 5.0)]  # Added bounds for price_ratio
    elif optimize_classes.lower() == 'business':
        bounds = [(120, 800) for _ in days] + [(1.01, 5.0)]
    elif optimize_classes.lower() == 'economy':
        bounds = [(50, 400) for _ in days] + [(1.01, 5.0)]
    
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
            price_ratio = optimized_prices[-1]
            
            if optimize_classes.lower() == 'both':
                y_B_opt = optimized_prices[:len(days)]
                y_E_opt = optimized_prices[len(days):-1]
            elif optimize_classes.lower() == 'business':
                y_B_opt = optimized_prices[:-1]
                y_E_opt = y_E_initial
            elif optimize_classes.lower() == 'economy':
                y_B_opt = y_B_initial
                y_E_opt = optimized_prices[:-1]
            
            # Ensure price ratio constraint is met
            y_B_opt = np.maximum(y_B_opt, y_E_opt * price_ratio)
            
            # Calculate expected sales and revenue
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
                'y_E_opt': y_E_opt,
                'price_ratio': price_ratio
            }
        else:
            # Return NaNs if optimization failed
            return {
                'total_revenue': np.nan,
                'total_sales_B': np.nan,
                'total_sales_E': np.nan,
                'y_B_opt': np.full_like(days, np.nan),
                'y_E_opt': np.full_like(days, np.nan),
                'price_ratio': np.nan
            }
    except Exception as e:
        # Handle unexpected errors
        print(f"Optimization Error: {e}")
        return {
            'total_revenue': np.nan,
            'total_sales_B': np.nan,
            'total_sales_E': np.nan,
            'y_B_opt': np.full_like(days, np.nan),
            'y_E_opt': np.full_like(days, np.nan),
            'price_ratio': np.nan
        }

def visualize_optimization_results(result, optimize_classes='both', C_B=50, C_E=150):
    """
    Visualizes the optimization results.

    Parameters:
        result (dict): Dictionary containing optimization results.
        optimize_classes (str): Which classes were optimized ('business', 'economy', 'both').
    """
    days = np.arange(30, -1, -1)
    
    plt.figure(figsize=(14, 6))
    
    if optimize_classes.lower() in ['business', 'both']:
        plt.subplot(1, 2, 1)
        if optimize_classes.lower() == 'both':
            y_B_opt = result['y_B_opt']
        else:
            y_B_opt = result['y_B_opt']
        plt.plot(days, y_B_opt, label='Business Class Price', color='blue')
        plt.xlabel('Days Before Departure')
        plt.ylabel('Price ($)')
        plt.title('Optimized Business Class Prices')
        plt.gca().invert_xaxis()
        plt.legend()
    
    if optimize_classes.lower() in ['economy', 'both']:
        plt.subplot(1, 2, 2)
        if optimize_classes.lower() == 'both':
            y_E_opt = result['y_E_opt']
        else:
            y_E_opt = result['y_E_opt']
        plt.plot(days, y_E_opt, label='Economy Class Price', color='green')
        plt.xlabel('Days Before Departure')
        plt.ylabel('Price ($)')
        plt.title('Optimized Economy Class Prices')
        plt.gca().invert_xaxis()
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"Total Revenue: ${result['total_revenue']:,.2f}")
    print(f"Total Business Tickets Sold: {result['total_sales_B']:.2f} / {C_B}")
    print(f"Total Economy Tickets Sold: {result['total_sales_E']:.2f} / {C_E}")
    print(f"Optimal Price Ratio (Business/Economy): {result['price_ratio']:.2f}")

def main():
    # Define parameter values (single set for simplicity)
    params = {
        'g_B': 10.83,      # Demand slope for Business
        'd_B': 12.9,       # Demand intercept for Business
        'h_B': 0.2,     # Demand decay for Business
        'a_B': 0.01,      # Probability slope for Business
        'b_B': 0.001,    # Probability decay for Business
        'g_E': 4.95,      # Demand slope for Economy
        'd_E': 13,      # Demand intercept for Economy
        'h_E': 0.117,     # Demand decay for Economy
        'a_E': 0.0067,     # Probability slope for Economy
        'b_E': 0.00291     # Probability decay for Economy
    }
    
    # Choose which classes to optimize: 'business', 'economy', 'both'
    optimize_classes = 'both'  # Change as needed
    
    # Capacities
    C = 200  # Total capacity
    C_B = 50  # Business class capacity
    C_E = C - C_B  # Economy class capacity
    
    # Run optimization with price ratio as a decision variable
    result = run_optimization(
        params=params,
        x_start=30,
        C=C,
        C_B=C_B,
        initial_prices_B=300.0,
        initial_prices_E=150.0,
        optimize_classes=optimize_classes,
        initial_price_ratio=1.5  # Initial guess for price ratio
    )
    
    # Check if optimization was successful
    if not np.isnan(result['total_revenue']):
        # Visualize results
        visualize_optimization_results(result, optimize_classes=optimize_classes, C_B=C_B, C_E=C_E)
    else:
        print("Optimization failed. Please check the parameters and constraints.")

if __name__ == "__main__":
    main()