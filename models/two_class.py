import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

class TwoClassModel:
    def __init__(self, params, x_start=30, C=200, C_B=50, initial_prices_B=300.0, initial_prices_E=150.0):
        self.params = params
        self.x_start = x_start
        self.C = C
        self.C_B = C_B
        self.C_E = C - C_B
        self.initial_prices_B = initial_prices_B
        self.initial_prices_E = initial_prices_E
        self.days = np.arange(x_start, -1, -1)

    def demand_B(self, x):
        return (self.params['g_B'] * x + self.params['d_B']) * np.exp(-self.params['h_B'] * x)
    
    def demand_E(self, x):
        return (self.params['g_E'] * x + self.params['d_E']) * np.exp(-self.params['h_E'] * x)
    
    def probability_B(self, y, x):
        return np.exp(-y * (self.params['a_B'] + self.params['b_B'] * x))
    
    def probability_E(self, y, x):
        return np.exp(-y * (self.params['a_E'] + self.params['b_E'] * x))

    def objective(self, prices):
        y_B = prices[:len(self.days)]
        y_E = prices[len(self.days):-1]
        price_ratio = prices[-1]
        
        y_B = np.maximum(y_B, y_E * price_ratio)

        revenue_B = np.sum(y_B * self.demand_B(self.days) * self.probability_B(y_B, self.days))
        revenue_E = np.sum(y_E * self.demand_E(self.days) * self.probability_E(y_E, self.days))
        revenue = revenue_B + revenue_E
        return -revenue

    def constraint_B(self, prices):
        y_B = prices[:len(self.days)]
        expected_sales_B = np.sum(self.demand_B(self.days) * self.probability_B(y_B, self.days))
        return self.C_B - expected_sales_B

    def constraint_E(self, prices):
        y_E = prices[len(self.days):-1]
        expected_sales_E = np.sum(self.demand_E(self.days) * self.probability_E(y_E, self.days))
        return self.C_E - expected_sales_E

    def constraint_price_ratio(self, prices):
        return prices[-1] - 1

    def optimize(self, initial_price_ratio=1.5):
        initial_prices = np.concatenate([
            np.full_like(self.days, self.initial_prices_B),
            np.full_like(self.days, self.initial_prices_E),
            [initial_price_ratio]
        ])
        
        bounds_B = [(120, 800) for _ in self.days]
        bounds_E = [(50, 400) for _ in self.days]
        bounds = bounds_B + bounds_E + [(1.01, 5.0)]

        constraints = [
            {'type': 'ineq', 'fun': self.constraint_B},
            {'type': 'ineq', 'fun': self.constraint_E},
            {'type': 'ineq', 'fun': self.constraint_price_ratio}
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
                price_ratio = optimized_prices[-1]
                y_B_opt = optimized_prices[:len(self.days)]
                y_E_opt = optimized_prices[len(self.days):-1]
                
                y_B_opt = np.maximum(y_B_opt, y_E_opt * price_ratio)
                
                expected_sales_B = self.demand_B(self.days) * self.probability_B(y_B_opt, self.days)
                expected_sales_E = self.demand_E(self.days) * self.probability_E(y_E_opt, self.days)
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
                print("Optimization failed. Please check the parameters and constraints.")
                return None
        except Exception as e:
            print(f"Optimization Error: {e}")
            return None

    def visualize_results(self, result):
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.days, result['y_B_opt'], label='Business Class Price', color='blue')
        plt.xlabel('Days Before Departure')
        plt.ylabel('Price ($)')
        plt.title('Optimized Business Class Prices')
        plt.gca().invert_xaxis()
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.days, result['y_E_opt'], label='Economy Class Price', color='green')
        plt.xlabel('Days Before Departure')
        plt.ylabel('Price ($)')
        plt.title('Optimized Economy Class Prices')
        plt.gca().invert_xaxis()
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"Total Revenue: ${result['total_revenue']:,.2f}")
        print(f"Total Business Tickets Sold: {result['total_sales_B']:.2f} / {self.C_B}")
        print(f"Total Economy Tickets Sold: {result['total_sales_E']:.2f} / {self.C_E}")
        print(f"Optimal Price Ratio (Business/Economy): {result['price_ratio']:.2f}")

def main():
    params = {
        'g_B': 10.83, 'd_B': 12.9, 'h_B': 0.2, 'a_B': 0.01, 'b_B': 0.001,
        'g_E': 4.95, 'd_E': 13, 'h_E': 0.117, 'a_E': 0.0067, 'b_E': 0.00291
    }
    
    model = TwoClassModel(params)
    result = model.optimize()
    
    if result:
        model.visualize_results(result)

if __name__ == "__main__":
    main()