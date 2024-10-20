# plot_simulations.py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.two_class_simple import TwoClassAirlineRevenueOptimizer

def run_simulation(params_economy, params_business, capacity_economy, capacity_business, x_start):
    optimizer = TwoClassAirlineRevenueOptimizer(
        params_economy=params_economy,
        params_business=params_business,
        capacity_economy=capacity_economy,
        capacity_business=capacity_business,
        x_start=x_start
    )
    
    try:
        optimal_y_economy, optimal_y_business, total_revenue, success = optimizer.optimize()
    except ValueError as e:
        print(e)
        return None

    days, economy_prices, business_prices, economy_demand, business_demand = optimizer.get_prices_and_demand(
        optimal_y_economy, optimal_y_business
    )

    return {
        'days': days,
        'economy_prices': economy_prices,
        'business_prices': business_prices,
        'economy_demand': economy_demand,
        'business_demand': business_demand,
        'total_revenue': total_revenue,
        'success': success
    }

def vary_parameters_and_plot():
    # Base parameters
    base_params_economy = (0.00667, 0.00291, 4.95, 13.0, 0.117)
    base_params_business = (0.00334, 0.00146, 2.48, 6.5, 0.0585)
    
    capacity_economy = 150
    capacity_business = 30
    x_start = 90  # Days before departure

    # Define realistic variations for economy and business classes
    variations = {
        'Economy_g': [4.5, 4.95, 5.4],      # Varying 'g' for economy
        'Economy_h': [0.11, 0.117, 0.124], # Varying 'h' for economy
        'Business_g': [2.3, 2.48, 2.65],    # Varying 'g' for business
        'Business_h': [0.055, 0.0585, 0.062]# Varying 'h' for business
    }

    # Initialize plot for Pricing Strategies
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: Pricing Strategies
    plt.subplot(2, 1, 1)
    plt.title('Optimal Pricing Strategies with Varying Parameters', fontsize=16)
    
    # Economy Class: Varying 'g'
    for g_e in variations['Economy_g']:
        params_economy = list(base_params_economy)
        params_economy[2] = g_e
        result = run_simulation(
            params_economy=params_economy,
            params_business=base_params_business,
            capacity_economy=capacity_economy,
            capacity_business=capacity_business,
            x_start=x_start
        )
        if result:
            plt.plot(result['days'], result['economy_prices'], label=f'Economy g={g_e}')
    
    # Economy Class: Varying 'h'
    for h_e in variations['Economy_h']:
        params_economy = list(base_params_economy)
        params_economy[4] = h_e
        result = run_simulation(
            params_economy=params_economy,
            params_business=base_params_business,
            capacity_economy=capacity_economy,
            capacity_business=capacity_business,
            x_start=x_start
        )
        if result:
            plt.plot(result['days'], result['economy_prices'], linestyle='--', label=f'Economy h={h_e}')
    
    # Business Class: Varying 'g'
    for g_b in variations['Business_g']:
        params_business = list(base_params_business)
        params_business[2] = g_b
        result = run_simulation(
            params_economy=base_params_economy,
            params_business=params_business,
            capacity_economy=capacity_economy,
            capacity_business=capacity_business,
            x_start=x_start
        )
        if result:
            plt.plot(result['days'], result['business_prices'], label=f'Business g={g_b}', color='orange')
    
    # Business Class: Varying 'h'
    for h_b in variations['Business_h']:
        params_business = list(base_params_business)
        params_business[4] = h_b
        result = run_simulation(
            params_economy=base_params_economy,
            params_business=params_business,
            capacity_economy=capacity_economy,
            capacity_business=capacity_business,
            x_start=x_start
        )
        if result:
            plt.plot(result['days'], result['business_prices'], linestyle='--', label=f'Business h={h_b}', color='orange')
    
    plt.xlabel('Days before departure', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.gca().invert_xaxis()  # Invert x-axis
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust layout to accommodate legend

    # Subplot 2: Demand Curves
    plt.subplot(2, 1, 2)
    
    plt.title('Demand Curves with Varying Parameters', fontsize=16)
    
    # Economy Class: Varying 'g'
    for g_e in variations['Economy_g']:
        params_economy = list(base_params_economy)
        params_economy[2] = g_e
        result = run_simulation(
            params_economy=params_economy,
            params_business=base_params_business,
            capacity_economy=capacity_economy,
            capacity_business=capacity_business,
            x_start=x_start
        )
        if result:
            plt.plot(result['days'], result['economy_demand'], label=f'Economy g={g_e}')
    
    # Economy Class: Varying 'h'
    for h_e in variations['Economy_h']:
        params_economy = list(base_params_economy)
        params_economy[4] = h_e
        result = run_simulation(
            params_economy=params_economy,
            params_business=base_params_business,
            capacity_economy=capacity_economy,
            capacity_business=capacity_business,
            x_start=x_start
        )
        if result:
            plt.plot(result['days'], result['economy_demand'], linestyle='--', label=f'Economy h={h_e}')
    
    # Business Class: Varying 'g'
    for g_b in variations['Business_g']:
        params_business = list(base_params_business)
        params_business[2] = g_b
        result = run_simulation(
            params_economy=base_params_economy,
            params_business=params_business,
            capacity_economy=capacity_economy,
            capacity_business=capacity_business,
            x_start=x_start
        )
        if result:
            plt.plot(result['days'], result['business_demand'], label=f'Business g={g_b}', color='orange')
    
    # Business Class: Varying 'h'
    for h_b in variations['Business_h']:
        params_business = list(base_params_business)
        params_business[4] = h_b
        result = run_simulation(
            params_economy=base_params_economy,
            params_business=params_business,
            capacity_economy=capacity_economy,
            capacity_business=capacity_business,
            x_start=x_start
        )
        if result:
            plt.plot(result['days'], result['business_demand'], linestyle='--', label=f'Business h={h_b}', color='orange')
    
    plt.xlabel('Days before departure', fontsize=12)
    plt.ylabel('Demand', fontsize=12)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.gca().invert_xaxis()  # Invert x-axis
    plt.tight_layout(rect=[0, 0, 0.75, 0.95])  # Adjust layout to accommodate legend and titles

    plt.show()

def main():
    vary_parameters_and_plot()

if __name__ == "__main__":
    main()
