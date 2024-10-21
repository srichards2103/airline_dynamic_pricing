import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars
import warnings

# Suppress warnings from scipy.integrate.quad when y(x) leads to underflow
warnings.filterwarnings("ignore", category=RuntimeWarning)

class TwoClassAirlineRevenueOptimizer:
    def __init__(self, params_economy, params_business, capacity_economy, capacity_business, x_start):
        """
        Initializes the optimizer with parameters for economy and business classes.

        Parameters:
        - params_economy: Tuple containing (a, b, g, d, h) for economy class.
        - params_business: Tuple containing (a, b, g, d, h) for business class.
        - capacity_economy: Total seats available for economy class.
        - capacity_business: Total seats available for business class.
        - x_start: Total number of days in the selling period.
        """
        self.params_economy = params_economy  # (a, b, g, d, h)
        self.params_business = params_business  # (a, b, g, d, h)
        self.capacity_economy = capacity_economy
        self.capacity_business = capacity_business
        self.x_start = x_start

    def f(self, x, params):
        """
        Potential demand function f(x).

        Parameters:
        - x: Days left until departure.
        - params: Tuple containing (a, b, g, d, h).

        Returns:
        - Potential demand at day x.
        """
        g, d, h = params[2], params[3], params[4]
        return (g * x + d) * np.exp(-h * x)

    def p(self, x, y, params):
        """
        Probability of purchase function p(x, y(x)).

        Parameters:
        - x: Days left until departure.
        - y: Price at day x.
        - params: Tuple containing (a, b, g, d, h).

        Returns:
        - Probability of purchase at day x with price y.
        """
        a, b = params[0], params[1]
        return np.exp(-y * (a + b * x))

    def revenue_integrand(self, x, y, params):
        """
        Integrand for calculating total revenue.

        Parameters:
        - x: Days left until departure.
        - y: Price function y(x).
        - params: Tuple containing (a, b, g, d, h).

        Returns:
        - Revenue contribution at day x.
        """
        return y(x) * self.f(x, params) * self.p(x, y(x), params)

    def capacity_integrand(self, x, y, params):
        """
        Integrand for checking capacity constraints.

        Parameters:
        - x: Days left until departure.
        - y: Price function y(x).
        - params: Tuple containing (a, b, g, d, h).

        Returns:
        - Demand contribution at day x.
        """
        return self.f(x, params) * self.p(x, y(x), params)

    def calculate_demand(self, x, y, params):
        """
        Calculates expected demand at day x with price y.

        Parameters:
        - x: Days left until departure.
        - y: Price at day x.
        - params: Tuple containing (a, b, g, d, h).

        Returns:
        - Expected demand.
        """
        return self.f(x, params) * self.p(x, y, params)

    def optimize_prices(self, remaining_capacity_economy, remaining_capacity_business):
        """
        Optimizes the price structure based on the remaining capacity.

        Parameters:
        - remaining_capacity_economy: Remaining seats in economy.
        - remaining_capacity_business: Remaining seats in business.

        Returns:
        - optimal_y_economy: Optimal price function for economy class.
        - optimal_y_business: Optimal price function for business class.
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

        return optimal_y_economy, optimal_y_business

    def simulate_bookings(self, y_economy, y_business, day, remaining_capacity_economy, remaining_capacity_business):
        """
        Simulates bookings for a given day based on current price functions.

        Parameters:
        - y_economy: Current price function for economy class.
        - y_business: Current price function for business class.
        - day: Current day (days left until departure).
        - remaining_capacity_economy: Remaining seats in economy.
        - remaining_capacity_business: Remaining seats in business.

        Returns:
        - bookings_econ: Number of bookings for economy class.
        - bookings_bus: Number of bookings for business class.
        - updated_remaining_capacity_economy: Updated remaining seats in economy.
        - updated_remaining_capacity_business: Updated remaining seats in business.
        """
        # Calculate lambda for economy and business
        lambda_economy = self.calculate_demand(day, y_economy(day), self.params_economy)
        lambda_business = self.calculate_demand(day, y_business(day), self.params_business)

        # Simulate bookings using Poisson distribution
        bookings_econ = np.random.poisson(lam=lambda_economy)
        bookings_bus = np.random.poisson(lam=lambda_business)

        # Ensure bookings do not exceed remaining capacity
        bookings_econ = min(bookings_econ, remaining_capacity_economy)
        bookings_bus = min(bookings_bus, remaining_capacity_business)

        # Update remaining capacities
        updated_remaining_capacity_economy = remaining_capacity_economy - bookings_econ
        updated_remaining_capacity_business = remaining_capacity_business - bookings_bus

        return bookings_econ, bookings_bus, updated_remaining_capacity_economy, updated_remaining_capacity_business

    def total_revenue(self, y_economy, y_business):
        """
        Calculates total expected revenue for both classes.

        Parameters:
        - y_economy: Price function for economy class.
        - y_business: Price function for business class.

        Returns:
        - Total expected revenue.
        """
        # Integration over the entire selling period is no longer needed
        # Revenue is calculated based on actual bookings
        # Therefore, this function is obsolete and can be removed or kept for analytical purposes
        # Here, we'll keep it but ensure it's not used in the simulation
        revenue_economy = quad(self.revenue_integrand, 0, self.x_start, args=(y_economy, self.params_economy))[0]
        revenue_business = quad(self.revenue_integrand, 0, self.x_start, args=(y_business, self.params_business))[0]
        return revenue_economy + revenue_business

    def capacity_constraint_economy(self, y_economy, remaining_capacity):
        """
        Capacity constraint for economy class.

        Parameters:
        - y_economy: Price function for economy class.
        - remaining_capacity: Remaining seats in economy.

        Returns:
        - Remaining capacity minus expected bookings.
        """
        expected_bookings = quad(self.capacity_integrand, 0, self.x_start, args=(y_economy, self.params_economy))[0]
        return remaining_capacity - expected_bookings

    def capacity_constraint_business(self, y_business, remaining_capacity):
        """
        Capacity constraint for business class.

        Parameters:
        - y_business: Price function for business class.
        - remaining_capacity: Remaining seats in business.

        Returns:
        - Remaining capacity minus expected bookings.
        """
        expected_bookings = quad(self.capacity_integrand, 0, self.x_start, args=(y_business, self.params_business))[0]
        return remaining_capacity - expected_bookings

def run_single_simulation(params_economy, params_business, capacity_economy, capacity_business, x_start):
    """
    Runs a single simulation of the airline revenue optimization.

    Returns:
    - total_revenue: Total revenue from the simulation.
    - total_bookings_economy: Total bookings for economy class.
    - total_bookings_business: Total bookings for business class.
    - bookings_economy: Array of daily bookings for economy class.
    - bookings_business: Array of daily bookings for business class.
    - prices_economy: Array of daily prices for economy class.
    - prices_business: Array of daily prices for business class.
    """
    optimizer = TwoClassAirlineRevenueOptimizer(
        params_economy=params_economy,
        params_business=params_business,
        capacity_economy=capacity_economy,
        capacity_business=capacity_business,
        x_start=x_start
    )

    remaining_capacity_economy = capacity_economy
    remaining_capacity_business = capacity_business

    bookings_economy = np.zeros(x_start, dtype=int)
    bookings_business = np.zeros(x_start, dtype=int)
    prices_economy = []
    prices_business = []

    total_revenue = 0

    for day in range(x_start, 0, -1):
        # Optimize prices based on current remaining capacity
        try:
            y_economy_opt, y_business_opt = optimizer.optimize_prices(
                remaining_capacity_economy,
                remaining_capacity_business
            )
        except ValueError as e:
            print(f"Optimization failed on day {day}: {e}")
            break

        # Simulate bookings for the current day
        bookings_econ, bookings_bus, remaining_capacity_economy, remaining_capacity_business = optimizer.simulate_bookings(
            y_economy_opt,
            y_business_opt,
            day,
            remaining_capacity_economy,
            remaining_capacity_business
        )

        # Record bookings
        bookings_economy[x_start - day] = bookings_econ
        bookings_business[x_start - day] = bookings_bus

        # Record prices
        price_econ = y_economy_opt(day)
        price_bus = y_business_opt(day)
        prices_economy.append(price_econ)
        prices_business.append(price_bus)

        # Accumulate revenue based on actual bookings
        revenue_today = bookings_econ * price_econ + bookings_bus * price_bus
        total_revenue += revenue_today

        # If all seats are sold, exit early
        if remaining_capacity_economy <= 0 and remaining_capacity_business <= 0:
            break

    total_bookings_economy = bookings_economy.sum()
    total_bookings_business = bookings_business.sum()

    return (
        total_revenue,
        total_bookings_economy,
        total_bookings_business,
        bookings_economy,
        bookings_business,
        prices_economy,
        prices_business
    )

def run_single_simulation_uniform_pricing(params_economy, params_business, capacity_economy, capacity_business, x_start, price_economy_fixed, price_business_fixed):
    """
    Runs a single simulation of the airline revenue under uniform pricing.

    Returns:
    - total_revenue: Total revenue from the simulation.
    - total_bookings_economy: Total bookings for economy class.
    - total_bookings_business: Total bookings for business class.
    - bookings_economy: Array of daily bookings for economy class.
    - bookings_business: Array of daily bookings for business class.
    """
    optimizer = TwoClassAirlineRevenueOptimizer(
        params_economy=params_economy,
        params_business=params_business,
        capacity_economy=capacity_economy,
        capacity_business=capacity_business,
        x_start=x_start
    )

    remaining_capacity_economy = capacity_economy
    remaining_capacity_business = capacity_business

    bookings_economy = np.zeros(x_start, dtype=int)
    bookings_business = np.zeros(x_start, dtype=int)
    prices_economy = []
    prices_business = []

    total_revenue = 0

    for day in range(x_start, 0, -1):
        # Use fixed prices
        price_econ = price_economy_fixed
        price_bus = price_business_fixed

        # Calculate expected demand
        lambda_economy = optimizer.calculate_demand(day, price_econ, optimizer.params_economy)
        lambda_business = optimizer.calculate_demand(day, price_bus, optimizer.params_business)

        # Simulate bookings using Poisson distribution
        bookings_econ = np.random.poisson(lam=lambda_economy)
        bookings_bus = np.random.poisson(lam=lambda_business)

        # Ensure bookings do not exceed remaining capacity
        bookings_econ = min(bookings_econ, remaining_capacity_economy)
        bookings_bus = min(bookings_bus, remaining_capacity_business)

        # Record bookings
        bookings_economy[x_start - day] = bookings_econ
        bookings_business[x_start - day] = bookings_bus

        # Record prices
        prices_economy.append(price_econ)
        prices_business.append(price_bus)

        # Accumulate revenue
        revenue_today = bookings_econ * price_econ + bookings_bus * price_bus
        total_revenue += revenue_today

        # Update remaining capacities
        remaining_capacity_economy -= bookings_econ
        remaining_capacity_business -= bookings_bus

        # If all seats are sold, exit early
        if remaining_capacity_economy <= 0 and remaining_capacity_business <= 0:
            break

    total_bookings_economy = bookings_economy.sum()
    total_bookings_business = bookings_business.sum()

    return (
        total_revenue,
        total_bookings_economy,
        total_bookings_business,
        bookings_economy,
        bookings_business,
        prices_economy,
        prices_business
    )

def run_monte_carlo_simulations(
    num_simulations,
    params_economy,
    params_business,
    capacity_economy,
    capacity_business,
    x_start
):
    """
    Runs multiple simulations and collects results.

    Returns:
    - revenues: List of total revenues from each simulation.
    - bookings_econ_list: List of total economy bookings from each simulation.
    - bookings_bus_list: List of total business bookings from each simulation.
    - all_bookings_economy: List of daily economy bookings arrays.
    - all_bookings_business: List of daily business bookings arrays.
    - all_prices_economy: List of daily economy prices arrays.
    - all_prices_business: List of daily business prices arrays.
    """
    revenues = []
    bookings_econ_list = []
    bookings_bus_list = []
    all_bookings_economy = []
    all_bookings_business = []
    all_prices_economy = []
    all_prices_business = []

    for _ in tqdm(range(num_simulations), desc="Running Dynamic Pricing Simulations"):
        (
            total_revenue,
            total_bookings_economy,
            total_bookings_business,
            bookings_economy,
            bookings_business,
            prices_economy,
            prices_business
        ) = run_single_simulation(
            params_economy,
            params_business,
            capacity_economy,
            capacity_business,
            x_start
        )

        revenues.append(total_revenue)
        bookings_econ_list.append(total_bookings_economy)
        bookings_bus_list.append(total_bookings_business)
        all_bookings_economy.append(bookings_economy)
        all_bookings_business.append(bookings_business)
        all_prices_economy.append(prices_economy)
        all_prices_business.append(prices_business)

    return (
        revenues,
        bookings_econ_list,
        bookings_bus_list,
        all_bookings_economy,
        all_bookings_business,
        all_prices_economy,
        all_prices_business
    )

def run_monte_carlo_simulations_uniform_pricing(
    num_simulations,
    params_economy,
    params_business,
    capacity_economy,
    capacity_business,
    x_start,
    price_economy_fixed,
    price_business_fixed
):
    """
    Runs multiple simulations under uniform pricing and collects results.

    Returns:
    - revenues: List of total revenues from each simulation.
    - bookings_econ_list: List of total economy bookings from each simulation.
    - bookings_bus_list: List of total business bookings from each simulation.
    - all_bookings_economy: List of daily economy bookings arrays.
    - all_bookings_business: List of daily business bookings arrays.
    - all_prices_economy: List of daily economy prices arrays.
    - all_prices_business: List of daily business prices arrays.
    """
    revenues = []
    bookings_econ_list = []
    bookings_bus_list = []
    all_bookings_economy = []
    all_bookings_business = []
    all_prices_economy = []
    all_prices_business = []

    for _ in tqdm(range(num_simulations), desc="Running Uniform Pricing Simulations"):
        (
            total_revenue,
            total_bookings_economy,
            total_bookings_business,
            bookings_economy,
            bookings_business,
            prices_economy,
            prices_business
        ) = run_single_simulation_uniform_pricing(
            params_economy,
            params_business,
            capacity_economy,
            capacity_business,
            x_start,
            price_economy_fixed,
            price_business_fixed
        )

        revenues.append(total_revenue)
        bookings_econ_list.append(total_bookings_economy)
        bookings_bus_list.append(total_bookings_business)
        all_bookings_economy.append(bookings_economy)
        all_bookings_business.append(bookings_business)
        all_prices_economy.append(prices_economy)
        all_prices_business.append(prices_business)

    return (
        revenues,
        bookings_econ_list,
        bookings_bus_list,
        all_bookings_economy,
        all_bookings_business,
        all_prices_economy,
        all_prices_business
    )

def plot_comparison_results(
    revenues_dynamic,
    revenues_uniform,
    bookings_econ_list_dynamic,
    bookings_econ_list_uniform,
    bookings_bus_list_dynamic,
    bookings_bus_list_uniform,
    all_bookings_economy_dynamic,
    all_bookings_economy_uniform,
    all_bookings_business_dynamic,
    all_bookings_business_uniform,
    all_prices_economy_dynamic,
    all_prices_economy_uniform,
    all_prices_business_dynamic,
    all_prices_business_uniform,
    x_start,
    num_simulations
):
    """
    Plots the comparison results between dynamic and uniform pricing strategies.
    """
    # Histogram of Total Revenues
    plt.figure(figsize=(14, 6))
    plt.hist(revenues_dynamic, bins=50, color='green', alpha=0.7, label='Dynamic Pricing')
    plt.hist(revenues_uniform, bins=50, color='blue', alpha=0.7, label='Uniform Pricing')
    plt.axvline(np.mean(revenues_dynamic), color='green', linestyle='dashed', linewidth=1, label=f"Mean Dynamic: ${np.mean(revenues_dynamic):.2f}")
    plt.axvline(np.mean(revenues_uniform), color='blue', linestyle='dashed', linewidth=1, label=f"Mean Uniform: ${np.mean(revenues_uniform):.2f}")
    plt.title('Comparison of Total Revenues')
    plt.xlabel('Total Revenue ($)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Histogram of Total Bookings - Economy
    plt.figure(figsize=(14, 6))
    plt.hist(bookings_econ_list_dynamic, bins=50, color='green', alpha=0.7, label='Dynamic Pricing')
    plt.hist(bookings_econ_list_uniform, bins=50, color='blue', alpha=0.7, label='Uniform Pricing')
    plt.axvline(np.mean(bookings_econ_list_dynamic), color='green', linestyle='dashed', linewidth=1, label=f"Mean Dynamic: {np.mean(bookings_econ_list_dynamic):.2f}")
    plt.axvline(np.mean(bookings_econ_list_uniform), color='blue', linestyle='dashed', linewidth=1, label=f"Mean Uniform: {np.mean(bookings_econ_list_uniform):.2f}")
    plt.title('Comparison of Total Economy Bookings')
    plt.xlabel('Total Economy Bookings')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Histogram of Total Bookings - Business
    plt.figure(figsize=(14, 6))
    plt.hist(bookings_bus_list_dynamic, bins=50, color='green', alpha=0.7, label='Dynamic Pricing')
    plt.hist(bookings_bus_list_uniform, bins=50, color='blue', alpha=0.7, label='Uniform Pricing')
    plt.axvline(np.mean(bookings_bus_list_dynamic), color='green', linestyle='dashed', linewidth=1, label=f"Mean Dynamic: {np.mean(bookings_bus_list_dynamic):.2f}")
    plt.axvline(np.mean(bookings_bus_list_uniform), color='blue', linestyle='dashed', linewidth=1, label=f"Mean Uniform: {np.mean(bookings_bus_list_uniform):.2f}")
    plt.title('Comparison of Total Business Bookings')
    plt.xlabel('Total Business Bookings')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Average Bookings Over Time
    all_bookings_economy_dynamic = np.array(all_bookings_economy_dynamic)
    all_bookings_business_dynamic = np.array(all_bookings_business_dynamic)
    all_bookings_economy_uniform = np.array(all_bookings_economy_uniform)
    all_bookings_business_uniform = np.array(all_bookings_business_uniform)
    
    mean_bookings_econ_dynamic = np.mean(all_bookings_economy_dynamic, axis=0)
    mean_bookings_bus_dynamic = np.mean(all_bookings_business_dynamic, axis=0)
    mean_bookings_econ_uniform = np.mean(all_bookings_economy_uniform, axis=0)
    mean_bookings_bus_uniform = np.mean(all_bookings_business_uniform, axis=0)
    
    days = np.arange(x_start, 0, -1)
    
    plt.figure(figsize=(14, 7))
    plt.plot(days, mean_bookings_econ_dynamic, label='Mean Economy Bookings - Dynamic', color='green')
    plt.plot(days, mean_bookings_econ_uniform, label='Mean Economy Bookings - Uniform', color='blue')
    plt.xlabel('Days Before Departure')
    plt.ylabel('Number of Bookings')
    plt.title('Average Daily Economy Bookings Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(14, 7))
    plt.plot(days, mean_bookings_bus_dynamic, label='Mean Business Bookings - Dynamic', color='green')
    plt.plot(days, mean_bookings_bus_uniform, label='Mean Business Bookings - Uniform', color='blue')
    plt.xlabel('Days Before Departure')
    plt.ylabel('Number of Bookings')
    plt.title('Average Daily Business Bookings Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Average Prices Over Time
    all_prices_economy_dynamic = np.array(all_prices_economy_dynamic)
    all_prices_business_dynamic = np.array(all_prices_business_dynamic)
    all_prices_economy_uniform = np.array(all_prices_economy_uniform)
    all_prices_business_uniform = np.array(all_prices_business_uniform)
    
    mean_prices_econ_dynamic = np.mean(all_prices_economy_dynamic, axis=0)
    mean_prices_bus_dynamic = np.mean(all_prices_business_dynamic, axis=0)
    mean_prices_econ_uniform = np.mean(all_prices_economy_uniform, axis=0)
    mean_prices_bus_uniform = np.mean(all_prices_business_uniform, axis=0)
    
    plt.figure(figsize=(14, 7))
    plt.plot(days, mean_prices_econ_dynamic, label='Mean Economy Price - Dynamic', color='green')
    plt.plot(days, mean_prices_econ_uniform, label='Mean Economy Price - Uniform', color='blue')
    plt.xlabel('Days Before Departure')
    plt.ylabel('Price ($)')
    plt.title('Average Daily Economy Prices Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(14, 7))
    plt.plot(days, mean_prices_bus_dynamic, label='Mean Business Price - Dynamic', color='green')
    plt.plot(days, mean_prices_bus_uniform, label='Mean Business Price - Uniform', color='blue')
    plt.xlabel('Days Before Departure')
    plt.ylabel('Price ($)')
    plt.title('Average Daily Business Prices Comparison')
    plt.legend()
    plt.grid(True)
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
    x_start = 30  # Days before departure
    
    # Number of simulations
    num_simulations = 100
    
    # Fixed prices for uniform pricing
    price_economy_fixed = 70
    price_business_fixed = 150
    
    # Run Monte Carlo simulations for dynamic pricing
    revenues_dynamic, bookings_econ_list_dynamic, bookings_bus_list_dynamic, all_bookings_economy_dynamic, all_bookings_business_dynamic, all_prices_economy_dynamic, all_prices_business_dynamic = run_monte_carlo_simulations(
        num_simulations=num_simulations,
        params_economy=base_params_economy,
        params_business=base_params_business,
        capacity_economy=capacity_economy,
        capacity_business=capacity_business,
        x_start=x_start
    )
    
    # Run Monte Carlo simulations for uniform pricing
    revenues_uniform, bookings_econ_list_uniform, bookings_bus_list_uniform, all_bookings_economy_uniform, all_bookings_business_uniform, all_prices_economy_uniform, all_prices_business_uniform = run_monte_carlo_simulations_uniform_pricing(
        num_simulations=num_simulations,
        params_economy=base_params_economy,
        params_business=base_params_business,
        capacity_economy=capacity_economy,
        capacity_business=capacity_business,
        x_start=x_start,
        price_economy_fixed=price_economy_fixed,
        price_business_fixed=price_business_fixed
    )
    
    # Plot the comparison results
    plot_comparison_results(
        revenues_dynamic,
        revenues_uniform,
        bookings_econ_list_dynamic,
        bookings_econ_list_uniform,
        bookings_bus_list_dynamic,
        bookings_bus_list_uniform,
        all_bookings_economy_dynamic,
        all_bookings_economy_uniform,
        all_bookings_business_dynamic,
        all_bookings_business_uniform,
        all_prices_economy_dynamic,
        all_prices_economy_uniform,
        all_prices_business_dynamic,
        all_prices_business_uniform,
        x_start,
        num_simulations
    )
    
    # Summary Statistics
    print("Dynamic Pricing Results:")
    print(f"Total Simulations Run: {num_simulations}")
    print(f"Average Total Revenue: ${np.mean(revenues_dynamic):.2f}")
    print(f"Revenue 95% Confidence Interval: (${np.percentile(revenues_dynamic, 2.5):.2f}, ${np.percentile(revenues_dynamic, 97.5):.2f})")
    print(f"Average Total Economy Bookings: {np.mean(bookings_econ_list_dynamic):.2f}")
    print(f"Economy Bookings 95% Confidence Interval: ({np.percentile(bookings_econ_list_dynamic, 2.5):.0f}, {np.percentile(bookings_econ_list_dynamic, 97.5):.0f})")
    print(f"Average Total Business Bookings: {np.mean(bookings_bus_list_dynamic):.2f}")
    print(f"Business Bookings 95% Confidence Interval: ({np.percentile(bookings_bus_list_dynamic, 2.5):.0f}, {np.percentile(bookings_bus_list_dynamic, 97.5):.0f})")
    
    print("\nUniform Pricing Results:")
    print(f"Total Simulations Run: {num_simulations}")
    print(f"Average Total Revenue: ${np.mean(revenues_uniform):.2f}")
    print(f"Revenue 95% Confidence Interval: (${np.percentile(revenues_uniform, 2.5):.2f}, ${np.percentile(revenues_uniform, 97.5):.2f})")
    print(f"Average Total Economy Bookings: {np.mean(bookings_econ_list_uniform):.2f}")
    print(f"Economy Bookings 95% Confidence Interval: ({np.percentile(bookings_econ_list_uniform, 2.5):.0f}, {np.percentile(bookings_econ_list_uniform, 97.5):.0f})")
    print(f"Average Total Business Bookings: {np.mean(bookings_bus_list_uniform):.2f}")
    print(f"Business Bookings 95% Confidence Interval: ({np.percentile(bookings_bus_list_uniform, 2.5):.0f}, {np.percentile(bookings_bus_list_uniform, 97.5):.0f})")

if __name__ == "__main__":
    main()
