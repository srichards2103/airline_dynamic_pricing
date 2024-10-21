
# import numpy as np
# from scipy.optimize import minimize
# from scipy.integrate import quad
# import matplotlib.pyplot as plt

# class TwoClassAirlineRevenueOptimizer:
#     def __init__(self, params_economy, params_business, capacity_economy, capacity_business, x_start):
#         self.params_economy = params_economy  # (a, b, g, d, h)
#         self.params_business = params_business  # (a, b, g, d, h)
#         self.capacity_economy = capacity_economy
#         self.capacity_business = capacity_business
#         self.x_start = x_start

#     def f(self, x, params):
#         g, d, h = params[2], params[3], params[4]
#         return (g * x + d) * np.exp(-h * x)

#     def p(self, x, y, params):
#         a, b = params[0], params[1]
#         return np.exp(-y * (a + b * x))

#     def revenue_integrand(self, x, y, params):
#         return y(x) * self.f(x, params) * self.p(x, y(x), params)

#     def capacity_integrand(self, x, y, params):
#         return self.f(x, params) * self.p(x, y(x), params)

#     def total_revenue(self, y_economy, y_business):
#         revenue_economy = quad(self.revenue_integrand, 0, self.x_start, args=(y_economy, self.params_economy))[0]
#         revenue_business = quad(self.revenue_integrand, 0, self.x_start, args=(y_business, self.params_business))[0]
#         return revenue_economy + revenue_business

#     def capacity_constraint_economy(self, y_economy):
#         return self.capacity_economy - quad(self.capacity_integrand, 0, self.x_start, args=(y_economy, self.params_economy))[0]

#     def capacity_constraint_business(self, y_business):
#         return self.capacity_business - quad(self.capacity_integrand, 0, self.x_start, args=(y_business, self.params_business))[0]

#     def calculate_demand(self, x, y, params):
#         return self.f(x, params) * self.p(x, y, params)

#     def optimize(self):
#         def objective(lambdas):
#             lambda_economy, lambda_business = lambdas
#             y_economy = lambda x: 1 / (self.params_economy[0] + self.params_economy[1] * x) + lambda_economy
#             y_business = lambda x: 1 / (self.params_business[0] + self.params_business[1] * x) + lambda_business
#             return -self.total_revenue(y_economy, y_business)

#         def constraint_economy(lambdas):
#             lambda_economy = lambdas[0]
#             y_economy = lambda x: 1 / (self.params_economy[0] + self.params_economy[1] * x) + lambda_economy
#             return self.capacity_constraint_economy(y_economy)

#         def constraint_business(lambdas):
#             lambda_business = lambdas[1]
#             y_business = lambda x: 1 / (self.params_business[0] + self.params_business[1] * x) + lambda_business
#             return self.capacity_constraint_business(y_business)

#         constraints = (
#             {'type': 'ineq', 'fun': constraint_economy},
#             {'type': 'ineq', 'fun': constraint_business}
#         )
        
#         result = minimize(objective, [0, 0], method='SLSQP', constraints=constraints)

#         if not result.success:
#             raise ValueError(f"Optimization failed: {result.message}")

#         optimal_lambda_economy, optimal_lambda_business = result.x
#         print(optimal_lambda_business)
#         print(optimal_lambda_economy)
#         optimal_y_economy = lambda x: 1 / (self.params_economy[0] + self.params_economy[1] * x) + optimal_lambda_economy
#         optimal_y_business = lambda x: 1 / (self.params_business[0] + self.params_business[1] * x) + optimal_lambda_business

#         return optimal_y_economy, optimal_y_business, -result.fun, result.success

#     def get_prices_and_demand(self, optimal_y_economy, optimal_y_business):
#         days = np.linspace(0, self.x_start, 20)
#         economy_prices = [optimal_y_economy(day) for day in days]
#         business_prices = [optimal_y_business(day) for day in days]
        
#         economy_demand = [self.calculate_demand(day, optimal_y_economy(day), self.params_economy) for day in days]
#         business_demand = [self.calculate_demand(day, optimal_y_business(day), self.params_business) for day in days]

#         return days, economy_prices, business_prices, economy_demand, business_demand

#     def plot_results(self, days, economy_prices, business_prices, economy_demand, business_demand, title_suffix=""):
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

#         # Price plot
#         ax1.plot(days, economy_prices, label='Economy Class', color='blue')
#         ax1.plot(days, business_prices, label='Business Class', color='red')
#         ax1.set_ylabel('Price ($)', fontsize=12)
#         ax1.set_title(f'Optimal Pricing Strategy {title_suffix}', fontsize=14)
#         ax1.legend(fontsize=12)
#         ax1.grid(True)

#         # Demand plot
#         ax2.plot(days, economy_demand, label='Economy Class', color='blue')
#         ax2.plot(days, business_demand, label='Business Class', color='red')
#         ax2.set_xlabel('Days before departure', fontsize=12)
#         ax2.set_ylabel('Demand', fontsize=12)
#         ax2.set_title(f'Demand Curves {title_suffix}', fontsize=14)
#         ax2.legend(fontsize=12)
#         ax2.grid(True)

       
        
        

#         # Invert x-axis for both subplots
#         ax1.invert_xaxis()
#         ax2.invert_xaxis()

#         plt.tight_layout()
#         plt.show()


# # Add this new class after the existing TwoClassAirlineRevenueOptimizer class

# def main():
#     base_params_economy = (0.00667, 0.00291, 4.95, 13.0, 0.117) # (a, b, g, d, h)
#     base_params_business = (0.00334, 0.00146, 2.48, 6.5, 0.0585) # (a, b, g, d, h)
#     capacity_economy = 150
#     capacity_business = 30
#     x_start = 20  # Days before departure
#     optimizer = TwoClassAirlineRevenueOptimizer(base_params_economy, base_params_business, capacity_economy, capacity_business, x_start)
#     optimal_y_economy, optimal_y_business, total_revenue, success = optimizer.optimize()
#     print(f"Total Revenue: {total_revenue}")
#     print(f"Optimization successful: {success}")
#     days, economy_prices, business_prices, economy_demand, business_demand = optimizer.get_prices_and_demand(optimal_y_economy, optimal_y_business)
#     optimizer.plot_results(days, economy_prices, business_prices, economy_demand, business_demand, title_suffix="")

# if __name__ == "__main__":
#     main()
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt

class TwoClassAirlineRevenueOptimizer:
    def __init__(self, params_economy, params_business, capacity_economy, capacity_business, x_start, N_simulations=1000, confidence_level=0.90):
        """
        Initializes the optimizer with parameters for economy and business classes.

        Parameters:
        - params_economy: Tuple containing (a, b, g, d, h) for economy class.
        - params_business: Tuple containing (a, b, g, d, h) for business class.
        - capacity_economy: Total seats available for economy class.
        - capacity_business: Total seats available for business class.
        - x_start: Total number of days in the selling period.
        - N_simulations: Number of simulation runs for confidence intervals.
        - confidence_level: Desired confidence interval (e.g., 0.90 for 90%).
        """
        self.params_economy = params_economy  # (a, b, g, d, h)
        self.params_business = params_business  # (a, b, g, d, h)
        self.capacity_economy = capacity_economy
        self.capacity_business = capacity_business
        self.x_start = x_start
        self.N_simulations = N_simulations
        self.confidence_level = confidence_level
        self.total_rev = 0

        # Initialize remaining capacities
        self.remaining_capacity_economy = capacity_economy
        self.remaining_capacity_business = capacity_business

        # Initialize booking records
        self.bookings_economy = np.zeros(x_start, dtype=int)
        self.bookings_business = np.zeros(x_start, dtype=int)

        # Initialize lists to store price and demand over time
        self.price_economy = []
        self.price_business = []
        self.demand_economy = []
        self.demand_business = []

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

    def capacity_constraint_economy(self, y_economy, remaining_capacity):
        return remaining_capacity - quad(self.capacity_integrand, 0, self.x_start, args=(y_economy, self.params_economy))[0]

    def capacity_constraint_business(self, y_business, remaining_capacity):
        return remaining_capacity - quad(self.capacity_integrand, 0, self.x_start, args=(y_business, self.params_business))[0]

    def calculate_demand(self, x, y, params):
        return self.f(x, params) * self.p(x, y, params)

    def optimize_prices(self, remaining_capacity_economy, remaining_capacity_business):
        """
        Optimizes the price structure based on the remaining capacity.

        Returns:
        - optimal_y_economy: Optimal price function for economy class.
        - optimal_y_business: Optimal price function for business class.
        - total_revenue: Total expected revenue.
        """
        def objective(lambdas):
            lambda_economy, lambda_business = lambdas
            # Define price functions with updated lambdas
            y_economy = lambda x: 1 / (self.params_economy[0] + self.params_economy[1] * x) + lambda_economy
            y_business = lambda x: 1 / (self.params_business[0] + self.params_business[1] * x) + lambda_business
            return -self.total_revenue(y_economy, y_business)

        def constraint_economy(lambdas):
            lambda_economy = lambdas[0]
            y_economy = lambda x: 1 / (self.params_economy[0] + self.params_economy[1] * x) + lambda_economy
            return self.capacity_constraint_economy(y_economy, remaining_capacity_economy)

        def constraint_business(lambdas):
            lambda_business = lambdas[1]
            y_business = lambda x: 1 / (self.params_business[0] + self.params_business[1] * x) + lambda_business
            return self.capacity_constraint_business(y_business, remaining_capacity_business)

        constraints = (
            {'type': 'ineq', 'fun': constraint_economy},
            {'type': 'ineq', 'fun': constraint_business}
        )

        initial_guess = [0, 0]  # Initial lambdas

        result = minimize(objective, initial_guess, method='SLSQP', constraints=constraints)

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        optimal_lambda_economy, optimal_lambda_business = result.x
        # Define the optimal price functions
        optimal_y_economy = lambda x: 1 / (self.params_economy[0] + self.params_economy[1] * x) + optimal_lambda_economy
        optimal_y_business = lambda x: 1 / (self.params_business[0] + self.params_business[1] * x) + optimal_lambda_business

        total_rev = -result.fun

        return optimal_y_economy, optimal_y_business, total_rev

    def simulate_bookings(self, y_economy, y_business, day):
        """
        Simulates bookings for a given day based on current price functions.

        Parameters:
        - y_economy: Current price function for economy class.
        - y_business: Current price function for business class.
        - day: Current day (days left until departure).

        Updates:
        - self.bookings_economy
        - self.bookings_business
        - self.remaining_capacity_economy
        - self.remaining_capacity_business
        """
        # Calculate lambda for economy and business
        lambda_economy = self.calculate_demand(day, y_economy(day), self.params_economy)
        lambda_business = self.calculate_demand(day, y_business(day), self.params_business)

        # Simulate bookings using Poisson distribution
        bookings_econ = np.random.poisson(lam=lambda_economy)
        bookings_bus = np.random.poisson(lam=lambda_business)

        # Ensure bookings do not exceed remaining capacity
        bookings_econ = min(bookings_econ, self.remaining_capacity_economy)
        bookings_bus = min(bookings_bus, self.remaining_capacity_business)

        # Update bookings and remaining capacity
        self.bookings_economy[self.x_start - day] = bookings_econ
        self.bookings_business[self.x_start - day] = bookings_bus
        self.remaining_capacity_economy -= bookings_econ
        self.remaining_capacity_business -= bookings_bus
        
        # Update total revenue
        self.total_rev += bookings_econ * y_economy(day) + bookings_bus * y_business(day)

        # Store current prices and demands
        self.price_economy.append(y_economy(day))
        self.price_business.append(y_business(day))
        self.demand_economy.append(lambda_economy)
        self.demand_business.append(lambda_business)

    def run_dynamic_optimization(self):
        """
        Runs the dynamic optimization process over the selling period.
        """
        for day in range(self.x_start, 0, -1):
            # Optimize prices based on current remaining capacity
            y_economy_opt, y_business_opt, total_rev = self.optimize_prices(self.remaining_capacity_economy, self.remaining_capacity_business)

            # Simulate bookings for the current day
            self.simulate_bookings(y_economy_opt, y_business_opt, day)

            # Check if capacities are exhausted
            if self.remaining_capacity_economy <= 0 and self.remaining_capacity_business <= 0:
                print(f"All seats sold by day {day}.")
                break

    def get_results(self):
        """
        Retrieves the booking and pricing results.

        Returns:
        - days: Array of days left until departure.
        - price_economy: List of economy class prices over time.
        - price_business: List of business class prices over time.
        - bookings_economy: Array of bookings per day for economy class.
        - bookings_business: Array of bookings per day for business class.
        """
        days = np.arange(self.x_start, 0, -1)
        return days, self.price_economy, self.price_business, self.bookings_economy, self.bookings_business, self.total_rev

    def plot_results(self, title_suffix=""):
        """
        Plots the price and booking results.

        Parameters:
        - title_suffix: Suffix for the plot titles.
        """
        days, price_econ, price_bus, bookings_econ, bookings_bus, total_rev = self.get_results()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

        # Price plot
        ax1.plot(days, price_econ, label='Economy Class Price', color='blue')
        ax1.plot(days, price_bus, label='Business Class Price', color='red')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title(f'Optimal Pricing Strategy {title_suffix}', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True)

        # Bookings plot
        ax2.bar(days - 0.2, bookings_econ, width=0.4, label='Economy Class Bookings', color='skyblue')
        ax2.bar(days + 0.2, bookings_bus, width=0.4, label='Business Class Bookings', color='salmon')
        ax2.set_xlabel('Days Before Departure', fontsize=12)
        ax2.set_ylabel('Number of Bookings', fontsize=12)
        ax2.set_title(f'Bookings Per Day {title_suffix}', fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True)

        # Invert x-axis to show days decreasing towards departure
        ax1.invert_xaxis()
        ax2.invert_xaxis()

        plt.tight_layout()
        plt.show()

def main():
    # Define model parameters (Best-fit values from the paper)
    # For Economy: (a, b, g, d, h)
    base_params_economy = (0.00667, 0.00291, 4.95, 13.0, 0.117)
    # For Business: (a, b, g, d, h) - assuming different parameters
    base_params_business = (0.00334, 0.00146, 2.48, 6.5, 0.0585)
    
    # Define capacities
    capacity_economy = 150
    capacity_business = 30
    
    # Selling period
    x_start = 20  # Days before departure
    
    # Initialize optimizer
    optimizer = TwoClassAirlineRevenueOptimizer(
        params_economy=base_params_economy,
        params_business=base_params_business,
        capacity_economy=capacity_economy,
        capacity_business=capacity_business,
        x_start=x_start,
        N_simulations=1000,
        confidence_level=0.90
    )
    
    # Run dynamic optimization
    optimizer.run_dynamic_optimization()
    
    # Retrieve and print results
    days, price_econ, price_bus, bookings_econ, bookings_bus, total_rev = optimizer.get_results()
    print("Day\tEconomy Price\tBusiness Price\tEconomy Bookings\tBusiness Bookings")
    for i in range(len(days)):
        print(f"{days[i]:>2}\t{price_econ[i]:>13.2f}\t{price_bus[i]:>14.2f}\t{bookings_econ[i]:>16}\t{bookings_bus[i]:>17}")
    
    # Plot the results
    optimizer.plot_results()

    # total bookings
    print(f"economy bookings: {sum(bookings_econ)}")
    print(f"business bookings: {sum(bookings_bus)}")
    print(f"total revenue: {total_rev}")

if __name__ == "__main__":
    main()
