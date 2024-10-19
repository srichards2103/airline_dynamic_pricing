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
        def objective(params):
            lambda_economy, lambda_business = params
            y_economy = lambda x: 1 / (self.params_economy[0] + self.params_economy[1] * x) + lambda_economy
            y_business = lambda x: 1 / (self.params_business[0] + self.params_business[1] * x) + lambda_business
            return -self.total_revenue(y_economy, y_business)

        def constraint_economy(params):
            lambda_economy = params[0]
            y_economy = lambda x: 1 / (self.params_economy[0] + self.params_economy[1] * x) + lambda_economy
            return self.capacity_constraint_economy(y_economy)

        def constraint_business(params):
            lambda_business = params[1]
            y_business = lambda x: 1 / (self.params_business[0] + self.params_business[1] * x) + lambda_business
            return self.capacity_constraint_business(y_business)

        constraints = (
            {'type': 'ineq', 'fun': constraint_economy},
            {'type': 'ineq', 'fun': constraint_business}
        )
        
        result = minimize(objective, [0, 0], method='SLSQP', constraints=constraints)

        optimal_lambda_economy, optimal_lambda_business = result.x
        optimal_y_economy = lambda x: 1 / (self.params_economy[0] + self.params_economy[1] * x) + optimal_lambda_economy
        optimal_y_business = lambda x: 1 / (self.params_business[0] + self.params_business[1] * x) + optimal_lambda_business

        return optimal_y_economy, optimal_y_business, -result.fun, result.success

# Example usage
if __name__ == "__main__":
    # Parameters for economy class (from the paper's fictional example)
    params_economy = (0.00667, 0.00291, 4.95, 13.0, 0.117)
    
    # Parameters for business class (hypothetical, assuming higher willingness to pay)
    params_business = (0.00334, 0.00146, 2.48, 6.5, 0.0585)
    
    capacity_economy = 150
    capacity_business = 30
    x_start = 90  # Assuming a 90-day selling period

    optimizer = TwoClassAirlineRevenueOptimizer(params_economy, params_business, capacity_economy, capacity_business, x_start)
    optimal_y_economy, optimal_y_business, total_revenue, optimization_success = optimizer.optimize()

    print(f"Optimal total revenue: ${total_revenue:.2f}")
    print(f"Optimization successful: {optimization_success}")

    # Print optimal prices for some example days
    for day in [90, 60, 30, 15, 7, 1]:
        print(f"Optimal economy price {day} days before departure: ${optimal_y_economy(day):.2f}")
        print(f"Optimal business price {day} days before departure: ${optimal_y_business(day):.2f}")

    # Plotting
    days = np.linspace(1, x_start, 100)
    economy_prices = [optimal_y_economy(day) for day in days]
    business_prices = [optimal_y_business(day) for day in days]
    
    economy_demand = [optimizer.calculate_demand(day, optimal_y_economy(day), optimizer.params_economy) for day in days]
    business_demand = [optimizer.calculate_demand(day, optimal_y_business(day), optimizer.params_business) for day in days]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Price plot
    ax1.plot(days, economy_prices, label='Economy Class')
    ax1.plot(days, business_prices, label='Business Class')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Optimal Pricing Strategy')
    ax1.legend()
    ax1.grid(True)

    # Demand plot
    ax2.plot(days, economy_demand, label='Economy Class')
    ax2.plot(days, business_demand, label='Business Class')
    ax2.set_xlabel('Days before departure')
    ax2.set_ylabel('Demand')
    ax2.set_title('Demand Curves')
    ax2.legend()
    ax2.grid(True)

    # Invert x-axis for both subplots
    ax1.invert_xaxis()
    ax2.invert_xaxis()

    plt.tight_layout()
    plt.show()
