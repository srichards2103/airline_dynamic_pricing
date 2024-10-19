import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
from models.one_class import OneClassModel
from models.two_class import TwoClassModel

warnings.filterwarnings("ignore")

def run_optimization(model_type, params, **kwargs):
    """
    Runs the optimization for a given model type and set of parameters.

    Parameters:
        model_type (str): 'one_class' or 'two_class'
        params (dict): Dictionary containing all necessary parameters.
        **kwargs: Additional arguments for model initialization.

    Returns:
        dict: Results containing total revenue, ticket sales, and optimized prices.
    """
    if model_type == 'one_class':
        model = OneClassModel(params, **kwargs)
    elif model_type == 'two_class':
        model = TwoClassModel(params, **kwargs)
    else:
        raise ValueError("Invalid model_type. Choose 'one_class' or 'two_class'.")

    result = model.optimize()
    return result, model

def visualize_optimization_results(result, model):
    """
    Visualizes the optimization results.

    Parameters:
        result (dict): Dictionary containing optimization results.
        model (OneClassModel or TwoClassModel): The model used for optimization.
    """
    model.visualize_results(result)

def main():
    # One-class model example
    params_one_class = {
        'g': 4.95, 'd': 13, 'h': 0.117, 'a': 0.0067, 'b': 0.00291
    }
    
    result_one_class, model_one_class = run_optimization('one_class', params_one_class, C=200, initial_price=200.0)
    
    if result_one_class:
        visualize_optimization_results(result_one_class, model_one_class)

    # Two-class model example
    params_two_class = {
        'g_B': 10.83, 'd_B': 12.9, 'h_B': 0.2, 'a_B': 0.01, 'b_B': 0.001,
        'g_E': 4.95, 'd_E': 13, 'h_E': 0.117, 'a_E': 0.0067, 'b_E': 0.00291
    }
    
    result_two_class, model_two_class = run_optimization('two_class', params_two_class, C=200, C_B=50, initial_prices_B=300.0, initial_prices_E=150.0)
    
    if result_two_class:
        visualize_optimization_results(result_two_class, model_two_class)

if __name__ == "__main__":
    main()
