
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt

class TwoClassAirlineRevenueOptimizer:
    def __init__(self, params_economy, params_business, capacity_economy, capacity_business, x_start):
        self.params_economy = params_economy  # (a, b, g, d, h)
        self.params_business = params_business  # (a, b, g, d, h)
        self.capacity_economy = capacity_economy
        self.capacity_business = capacity_business
        self.x_start = x_start

    def f(self, x, params):
        g, d, h = params[2], params[3], params[4]
        return (g * x + d) * np.exp(-h * x)

    def p(self, x, y, params):
        a, b = params[0], params[1]
        return np.exp(-y * (a + b * x))

    def revenue_integrand(self, x, y, params):
        return y(x) * self.f(x, params) * self.p(x, y(x), params)

    def capacity_integrand(self, x, y, params):
        return self.f(x, params) * self.p(x, y(x), params)

    def total_revenue(self, y_economy, y_business):
        revenue_economy = quad(self.revenue_integrand, 0, self.x_start, args=(y_economy, self.params_economy))[0]
        revenue_business = quad(self.revenue_integrand, 0, self.x_start, args=(y_business, self.params_business))[0]
        return revenue_economy + revenue_business

    def capacity_constraint_economy(self, y_economy):
        return self.capacity_economy - quad(self.capacity_integrand, 0, self.x_start, args=(y_economy, self.params_economy))[0]

    def capacity_constraint_business(self, y_business):
        return self.capacity_business - quad(self.capacity_integrand, 0, self.x_start, args=(y_business, self.params_business))[0]

    def calculate_demand(self, x, y, params):
        return self.f(x, params) * self.p(x, y, params)

    def optimize(self):
        def objective(lambdas):
            lambda_economy, lambda_business = lambdas
            y_economy = lambda x: 1 / (self.params_economy[0] + self.params_economy[1] * x) + lambda_economy
            y_business = lambda x: 1 / (self.params_business[0] + self.params_business[1] * x) + lambda_business
            return -self.total_revenue(y_economy, y_business)

        def constraint_economy(lambdas):
            lambda_economy = lambdas[0]
            y_economy = lambda x: 1 / (self.params_economy[0] + self.params_economy[1] * x) + lambda_economy
            return self.capacity_constraint_economy(y_economy)

        def constraint_business(lambdas):
            lambda_business = lambdas[1]
            y_business = lambda x: 1 / (self.params_business[0] + self.params_business[1] * x) + lambda_business
            return self.capacity_constraint_business(y_business)

        constraints = (
            {'type': 'ineq', 'fun': constraint_economy},
            {'type': 'ineq', 'fun': constraint_business}
        )
        
        result = minimize(objective, [0, 0], method='SLSQP', constraints=constraints)

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        optimal_lambda_economy, optimal_lambda_business = result.x
        print(optimal_lambda_business)
        print(optimal_lambda_economy)
        optimal_y_economy = lambda x: 1 / (self.params_economy[0] + self.params_economy[1] * x) + optimal_lambda_economy
        optimal_y_business = lambda x: 1 / (self.params_business[0] + self.params_business[1] * x) + optimal_lambda_business

        return optimal_y_economy, optimal_y_business, -result.fun, result.success

    def get_prices_and_demand(self, optimal_y_economy, optimal_y_business):
        days = np.linspace(0, self.x_start, 20)
        economy_prices = [optimal_y_economy(day) for day in days]
        business_prices = [optimal_y_business(day) for day in days]
        
        economy_demand = [self.calculate_demand(day, optimal_y_economy(day), self.params_economy) for day in days]
        business_demand = [self.calculate_demand(day, optimal_y_business(day), self.params_business) for day in days]

        return days, economy_prices, business_prices, economy_demand, business_demand

    def plot_results(self, days, economy_prices, business_prices, economy_demand, business_demand, title_suffix=""):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

        # Price plot
        ax1.plot(days, economy_prices, label='Economy Class', color='blue')
        ax1.plot(days, business_prices, label='Business Class', color='red')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title(f'Optimal Pricing Strategy {title_suffix}', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True)

        # Demand plot
        ax2.plot(days, economy_demand, label='Economy Class', color='blue')
        ax2.plot(days, business_demand, label='Business Class', color='red')
        ax2.set_xlabel('Days before departure', fontsize=12)
        ax2.set_ylabel('Demand', fontsize=12)
        ax2.set_title(f'Demand Curves {title_suffix}', fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True)

       
        
        

        # Invert x-axis for both subplots
        ax1.invert_xaxis()
        ax2.invert_xaxis()

        plt.tight_layout()
        plt.show()


# Add this new class after the existing TwoClassAirlineRevenueOptimizer class

def main():
    base_params_economy = (0.00667, 0.00291, 4.95, 13.0, 0.117) # (a, b, g, d, h)
    base_params_business = (0.00334, 0.00146, 2.48, 6.5, 0.0585) # (a, b, g, d, h)
    capacity_economy = 150
    capacity_business = 30
    x_start = 20  # Days before departure
    optimizer = TwoClassAirlineRevenueOptimizer(base_params_economy, base_params_business, capacity_economy, capacity_business, x_start)
    optimal_y_economy, optimal_y_business, total_revenue, success = optimizer.optimize()
    print(f"Total Revenue: {total_revenue}")
    print(f"Optimization successful: {success}")
    days, economy_prices, business_prices, economy_demand, business_demand = optimizer.get_prices_and_demand(optimal_y_economy, optimal_y_business)
    optimizer.plot_results(days, economy_prices, business_prices, economy_demand, business_demand, title_suffix="")

if __name__ == "__main__":
    main()