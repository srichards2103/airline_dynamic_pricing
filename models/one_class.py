import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

class OneClassModel:
    def __init__(self, params, x_start=30, C=200, initial_price=200.0):
        self.params = params
        self.x_start = x_start
        self.C = C
        self.initial_price = initial_price
        self.days = np.arange(x_start, -1, -1)

    def demand(self, x):
        return (self.params['g'] * x + self.params['d']) * np.exp(-self.params['h'] * x)
    
    def probability(self, y, x):
        return np.exp(-y * (self.params['a'] + self.params['b'] * x))

    def objective(self, prices):
        revenue = np.sum(prices * self.demand(self.days) * self.probability(prices, self.days))
        return -revenue

    def constraint_capacity(self, prices):
        expected_sales = np.sum(self.demand(self.days) * self.probability(prices, self.days))
        return self.C - expected_sales

    def optimize(self):
        initial_prices = np.full_like(self.days, self.initial_price)
        
        bounds = [(50, 800) for _ in self.days]

        constraints = [
            {'type': 'ineq', 'fun': self.constraint_capacity}
        ]

        try:
            result = minimize(
                self.objective,
                initial_prices,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'disp': False, 'maxiter': 1000}
            )
            
            if result.success:
                optimized_prices = result.x
                
                expected_sales = self.demand(self.days) * self.probability(optimized_prices, self.days)
                total_sales = np.sum(expected_sales)
                total_revenue = np.sum(optimized_prices * expected_sales)
                
                return {
                    'total_revenue': total_revenue,
                    'total_sales': total_sales,
                    'optimized_prices': optimized_prices
                }
            else:
                print("Optimization failed. Please check the parameters and constraints.")
                return None
        except Exception as e:
            print(f"Optimization Error: {e}")
            return None

    def visualize_results(self, result):
        plt.figure(figsize=(10, 6))
        
        plt.plot(self.days, result['optimized_prices'], label='Optimized Prices', color='blue')
        plt.xlabel('Days Before Departure')
        plt.ylabel('Price ($)')
        plt.title('Optimized Ticket Prices')
        plt.gca().invert_xaxis()
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"Total Revenue: ${result['total_revenue']:,.2f}")
        print(f"Total Tickets Sold: {result['total_sales']:.2f} / {self.C}")

def main():
    # Example usage for OneClassModel
    params_one_class = {
        'g': 4.95, 'd': 13, 'h': 0.117, 'a': 0.0067, 'b': 0.00291
    }
    
    one_class_model = OneClassModel(params_one_class, C=200)
    result_one_class = one_class_model.optimize()
    
    if result_one_class:
        one_class_model.visualize_results(result_one_class)


if __name__ == "__main__":
    main()