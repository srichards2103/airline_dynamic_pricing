import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars
import warnings

# Suppress warnings from scipy.optimize.minimize when optimization fails
warnings.filterwarnings("ignore", category=RuntimeWarning)

class TwoClassAirlineRevenueOptimizer:
    def __init__(self, params_economy, params_business, x_start):
        """
        Initializes the optimizer with parameters for economy and business classes.

        Parameters:
        - params_economy: Tuple containing (a, b, g, d, h) for economy class.
        - params_business: Tuple containing (a, b, g, d, h) for business class.
        - x_start: Total number of days in the selling period.
        """
        self.params_economy = params_economy  # (a, b, g, d, h)
        self.params_business = params_business  # (a, b, g, d, h)
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

    def optimize_prices(self, remaining_capacity_economy, remaining_capacity_business, day):
        """
        Optimizes the price structure based on the remaining capacity for the current day.

        Parameters:
        - remaining_capacity_economy: Remaining seats in economy.
        - remaining_capacity_business: Remaining seats in business.
        - day: Current day (days left until departure).

        Returns:
        - optimal_y_economy: Optimal price for economy class.
        - optimal_y_business: Optimal price for business class.
        """
        def objective(prices):
            y_economy, y_business = prices
            lambda_econ = self.calculate_demand(day, y_economy, self.params_economy)
            lambda_bus = self.calculate_demand(day, y_business, self.params_business)
            # Negative because we minimize
            return -(y_economy * lambda_econ + y_business * lambda_bus)

        def constraint_economy(prices):
            y_economy, _ = prices
            lambda_econ = self.calculate_demand(day, y_economy, self.params_economy)
            return remaining_capacity_economy - lambda_econ

        def constraint_business(prices):
            _, y_business = prices
            lambda_bus = self.calculate_demand(day, y_business, self.params_business)
            return remaining_capacity_business - lambda_bus

        constraints = (
            {'type': 'ineq', 'fun': constraint_economy},
            {'type': 'ineq', 'fun': constraint_business}
        )

        # Set bounds for prices to be positive
        bounds = [(1, None), (1, None)]  # Prices must be at least $1

        # Initial guess: $70 for both classes
        initial_guess = [70, 70]

        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        optimal_y_economy, optimal_y_business = result.x
        return optimal_y_economy, optimal_y_business

    def simulate_bookings(self, y_economy, y_business, day, remaining_capacity_economy, remaining_capacity_business):
        """
        Simulates bookings for a given day based on current price functions.

        Parameters:
        - y_economy: Current price for economy class.
        - y_business: Current price for business class.
        - day: Current day (days left until departure).
        - remaining_capacity_economy: Remaining seats in economy.
        - remaining_capacity_business: Remaining seats in business.

        Returns:
        - bookings_econ: Number of bookings for economy class.
        - bookings_bus: Number of bookings for business class.
        - updated_remaining_capacity_economy: Updated remaining seats in economy.
        - updated_remaining_capacity_business: Updated remaining seats in business.
        """
        # Calculate expected demand (lambda) for economy and business
        lambda_economy = self.calculate_demand(day, y_economy, self.params_economy)
        lambda_business = self.calculate_demand(day, y_business, self.params_business)

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

def run_uniform_simulation(params_economy, params_business, capacity_economy, capacity_business, x_start):
    """
    Runs a single simulation using Uniform Pricing.

    Parameters:
    - params_economy: Parameters for economy class.
    - params_business: Parameters for business class.
    - capacity_economy: Total seats in economy.
    - capacity_business: Total seats in business.
    - x_start: Total number of days in the selling period.

    Returns:
    - uniform_revenue: Total revenue from Uniform Pricing.
    - uniform_bookings_economy: Total economy bookings from Uniform Pricing.
    - uniform_bookings_business: Total business bookings from Uniform Pricing.
    - uniform_bookings_economy_daily: Array of daily economy bookings from Uniform Pricing.
    - uniform_bookings_business_daily: Array of daily business bookings from Uniform Pricing.
    - uniform_prices_economy_daily: Array of daily economy prices from Uniform Pricing.
    - uniform_prices_business_daily: Array of daily business prices from Uniform Pricing.
    """
    # Fixed price for uniform pricing
    fixed_price_economy = 70
    fixed_price_business = 70

    remaining_capacity_economy = capacity_economy
    remaining_capacity_business = capacity_business

    uniform_bookings_economy_daily = np.zeros(x_start, dtype=int)
    uniform_bookings_business_daily = np.zeros(x_start, dtype=int)
    uniform_prices_economy_daily = []
    uniform_prices_business_daily = []

    uniform_revenue = 0

    # Initialize optimizer to access demand functions
    optimizer = TwoClassAirlineRevenueOptimizer(
        params_economy=params_economy,
        params_business=params_business,
        x_start=x_start
    )

    for day in range(x_start, 0, -1):
        # Set fixed prices
        price_economy = fixed_price_economy
        price_business = fixed_price_business
        uniform_prices_economy_daily.append(price_economy)
        uniform_prices_business_daily.append(price_business)

        # Calculate expected demand
        lambda_econ = optimizer.calculate_demand(day, price_economy, params_economy)
        lambda_bus = optimizer.calculate_demand(day, price_business, params_business)

        # Simulate bookings using Poisson distribution
        bookings_econ = np.random.poisson(lam=lambda_econ)
        bookings_bus = np.random.poisson(lam=lambda_bus)

        # Ensure bookings do not exceed remaining capacity
        bookings_econ = min(bookings_econ, remaining_capacity_economy)
        bookings_bus = min(bookings_bus, remaining_capacity_business)

        # Update remaining capacities
        remaining_capacity_economy -= bookings_econ
        remaining_capacity_business -= bookings_bus

        # Record bookings
        uniform_bookings_economy_daily[x_start - day] = bookings_econ
        uniform_bookings_business_daily[x_start - day] = bookings_bus

        # Accumulate revenue
        uniform_revenue += bookings_econ * price_economy + bookings_bus * price_business

        # If all seats are sold, exit early
        if remaining_capacity_economy <= 0 and remaining_capacity_business <= 0:
            break

    uniform_bookings_economy = uniform_bookings_economy_daily.sum()
    uniform_bookings_business = uniform_bookings_business_daily.sum()

    return (
        uniform_revenue,
        uniform_bookings_economy,
        uniform_bookings_business,
        uniform_bookings_economy_daily,
        uniform_bookings_business_daily,
        uniform_prices_economy_daily,
        uniform_prices_business_daily
    )

def run_single_simulation(params_economy, params_business, capacity_economy, capacity_business, x_start):
    """
    Runs a single simulation for both Dynamic Pricing and Uniform Pricing.

    Returns:
    - dynamic_revenue: Total revenue from Dynamic Pricing.
    - dynamic_bookings_economy: Total economy bookings from Dynamic Pricing.
    - dynamic_bookings_business: Total business bookings from Dynamic Pricing.
    - dynamic_bookings_economy_daily: Array of daily economy bookings from Dynamic Pricing.
    - dynamic_bookings_business_daily: Array of daily business bookings from Dynamic Pricing.
    - dynamic_prices_economy_daily: Array of daily economy prices from Dynamic Pricing.
    - dynamic_prices_business_daily: Array of daily business prices from Dynamic Pricing.
    - uniform_revenue: Total revenue from Uniform Pricing.
    - uniform_bookings_economy: Total economy bookings from Uniform Pricing.
    - uniform_bookings_business: Total business bookings from Uniform Pricing.
    - uniform_bookings_economy_daily: Array of daily economy bookings from Uniform Pricing.
    - uniform_bookings_business_daily: Array of daily business bookings from Uniform Pricing.
    - uniform_prices_economy_daily: Array of daily economy prices from Uniform Pricing.
    - uniform_prices_business_daily: Array of daily business prices from Uniform Pricing.
    """
    # Dynamic Pricing Simulation
    dynamic_optimizer = TwoClassAirlineRevenueOptimizer(
        params_economy=params_economy,
        params_business=params_business,
        x_start=x_start
    )

    remaining_capacity_economy = capacity_economy
    remaining_capacity_business = capacity_business

    dynamic_bookings_economy_daily = np.zeros(x_start, dtype=int)
    dynamic_bookings_business_daily = np.zeros(x_start, dtype=int)
    dynamic_prices_economy_daily = []
    dynamic_prices_business_daily = []

    dynamic_revenue = 0

    for day in range(x_start, 0, -1):
        # Optimize prices based on current remaining capacity
        try:
            y_economy_opt, y_business_opt = dynamic_optimizer.optimize_prices(
                remaining_capacity_economy,
                remaining_capacity_business,
                day
            )
        except ValueError as e:
            print(f"Dynamic Optimization failed on day {day}: {e}")
            break

        # Simulate bookings for the current day
        bookings_econ, bookings_bus, remaining_capacity_economy, remaining_capacity_business = dynamic_optimizer.simulate_bookings(
            y_economy_opt,
            y_business_opt,
            day,
            remaining_capacity_economy,
            remaining_capacity_business
        )

        # Record bookings
        dynamic_bookings_economy_daily[x_start - day] = bookings_econ
        dynamic_bookings_business_daily[x_start - day] = bookings_bus

        # Record prices
        price_econ = y_economy_opt
        price_bus = y_business_opt
        dynamic_prices_economy_daily.append(price_econ)
        dynamic_prices_business_daily.append(price_bus)

        # Accumulate revenue based on actual bookings
        dynamic_revenue += bookings_econ * price_econ + bookings_bus * price_bus

        # If all seats are sold, exit early
        if remaining_capacity_economy <= 0 and remaining_capacity_business <= 0:
            break

    dynamic_bookings_economy = dynamic_bookings_economy_daily.sum()
    dynamic_bookings_business = dynamic_bookings_business_daily.sum()

    # Uniform Pricing Simulation
    uniform_revenue, uniform_bookings_economy, uniform_bookings_business, \
    uniform_bookings_economy_daily, uniform_bookings_business_daily, \
    uniform_prices_economy_daily, uniform_prices_business_daily = run_uniform_simulation(
        params_economy,
        params_business,
        capacity_economy,
        capacity_business,
        x_start
    )

    return (
        dynamic_revenue,
        dynamic_bookings_economy,
        dynamic_bookings_business,
        dynamic_bookings_economy_daily,
        dynamic_bookings_business_daily,
        dynamic_prices_economy_daily,
        dynamic_prices_business_daily,
        uniform_revenue,
        uniform_bookings_economy,
        uniform_bookings_business,
        uniform_bookings_economy_daily,
        uniform_bookings_business_daily,
        uniform_prices_economy_daily,
        uniform_prices_business_daily
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
    Runs multiple simulations and collects results for both Dynamic and Uniform Pricing.

    Returns:
    - dynamic_revenues: List of total revenues from Dynamic Pricing.
    - dynamic_bookings_econ_list: List of total economy bookings from Dynamic Pricing.
    - dynamic_bookings_bus_list: List of total business bookings from Dynamic Pricing.
    - dynamic_bookings_economy_daily_all: List of daily economy bookings arrays from Dynamic Pricing.
    - dynamic_bookings_business_daily_all: List of daily business bookings arrays from Dynamic Pricing.
    - dynamic_prices_economy_daily_all: List of daily economy prices arrays from Dynamic Pricing.
    - dynamic_prices_business_daily_all: List of daily business prices arrays from Dynamic Pricing.
    - uniform_revenues: List of total revenues from Uniform Pricing.
    - uniform_bookings_econ_list: List of total economy bookings from Uniform Pricing.
    - uniform_bookings_bus_list: List of total business bookings from Uniform Pricing.
    - uniform_bookings_economy_daily_all: List of daily economy bookings arrays from Uniform Pricing.
    - uniform_bookings_business_daily_all: List of daily business bookings arrays from Uniform Pricing.
    - uniform_prices_economy_daily_all: List of daily economy prices arrays from Uniform Pricing.
    - uniform_prices_business_daily_all: List of daily business prices arrays from Uniform Pricing.
    """
    dynamic_revenues = []
    dynamic_bookings_econ_list = []
    dynamic_bookings_bus_list = []
    dynamic_bookings_economy_daily_all = []
    dynamic_bookings_business_daily_all = []
    dynamic_prices_economy_daily_all = []
    dynamic_prices_business_daily_all = []

    uniform_revenues = []
    uniform_bookings_econ_list = []
    uniform_bookings_bus_list = []
    uniform_bookings_economy_daily_all = []
    uniform_bookings_business_daily_all = []
    uniform_prices_economy_daily_all = []
    uniform_prices_business_daily_all = []

    for _ in tqdm(range(num_simulations), desc="Running Simulations"):
        (
            dynamic_revenue,
            dynamic_bookings_economy,
            dynamic_bookings_business,
            dynamic_bookings_economy_daily,
            dynamic_bookings_business_daily,
            dynamic_prices_economy_daily,
            dynamic_prices_business_daily,
            uniform_revenue,
            uniform_bookings_economy,
            uniform_bookings_business,
            uniform_bookings_economy_daily,
            uniform_bookings_business_daily,
            uniform_prices_economy_daily,
            uniform_prices_business_daily
        ) = run_single_simulation(
            params_economy,
            params_business,
            capacity_economy,
            capacity_business,
            x_start
        )

        dynamic_revenues.append(dynamic_revenue)
        dynamic_bookings_econ_list.append(dynamic_bookings_economy)
        dynamic_bookings_bus_list.append(dynamic_bookings_business)
        dynamic_bookings_economy_daily_all.append(dynamic_bookings_economy_daily)
        dynamic_bookings_business_daily_all.append(dynamic_bookings_business_daily)
        dynamic_prices_economy_daily_all.append(dynamic_prices_economy_daily)
        dynamic_prices_business_daily_all.append(dynamic_prices_business_daily)

        uniform_revenues.append(uniform_revenue)
        uniform_bookings_econ_list.append(uniform_bookings_economy)
        uniform_bookings_bus_list.append(uniform_bookings_business)
        uniform_bookings_economy_daily_all.append(uniform_bookings_economy_daily)
        uniform_bookings_business_daily_all.append(uniform_bookings_business_daily)
        uniform_prices_economy_daily_all.append(uniform_prices_economy_daily)
        uniform_prices_business_daily_all.append(uniform_prices_business_daily)

    return (
        dynamic_revenues,
        dynamic_bookings_econ_list,
        dynamic_bookings_bus_list,
        dynamic_bookings_economy_daily_all,
        dynamic_bookings_business_daily_all,
        dynamic_prices_economy_daily_all,
        dynamic_prices_business_daily_all,
        uniform_revenues,
        uniform_bookings_econ_list,
        uniform_bookings_bus_list,
        uniform_bookings_economy_daily_all,
        uniform_bookings_business_daily_all,
        uniform_prices_economy_daily_all,
        uniform_prices_business_daily_all
    )

def plot_monte_carlo_results(
    dynamic_revenues,
    dynamic_bookings_econ_list,
    dynamic_bookings_bus_list,
    dynamic_bookings_economy_daily_all,
    dynamic_bookings_business_daily_all,
    dynamic_prices_economy_daily_all,
    dynamic_prices_business_daily_all,
    uniform_revenues,
    uniform_bookings_econ_list,
    uniform_bookings_bus_list,
    uniform_bookings_economy_daily_all,
    uniform_bookings_business_daily_all,
    uniform_prices_economy_daily_all,
    uniform_prices_business_daily_all,
    x_start,
    num_simulations
):
    """
    Generates Monte Carlo plots based on simulation results for both Dynamic and Uniform Pricing.
    """
    # Histogram of Total Revenues
    plt.figure(figsize=(14, 6))
    plt.hist(dynamic_revenues, bins=50, alpha=0.7, label='Dynamic Pricing', color='green')
    plt.hist(uniform_revenues, bins=50, alpha=0.7, label='Uniform Pricing', color='orange')
    plt.axvline(np.mean(dynamic_revenues), color='darkgreen', linestyle='dashed', linewidth=1, label=f"Dynamic Mean: ${np.mean(dynamic_revenues):.2f}")
    plt.axvline(np.mean(uniform_revenues), color='darkorange', linestyle='dashed', linewidth=1, label=f"Uniform Mean: ${np.mean(uniform_revenues):.2f}")
    plt.title('Monte Carlo Simulation of Total Revenues')
    plt.xlabel('Total Revenue ($)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Histogram of Total Bookings - Economy
    plt.figure(figsize=(14, 6))
    plt.hist(dynamic_bookings_econ_list, bins=50, alpha=0.7, label='Dynamic Pricing', color='blue')
    plt.hist(uniform_bookings_econ_list, bins=50, alpha=0.7, label='Uniform Pricing', color='skyblue')
    plt.axvline(np.mean(dynamic_bookings_econ_list), color='darkblue', linestyle='dashed', linewidth=1, label=f"Dynamic Mean: {np.mean(dynamic_bookings_econ_list):.2f}")
    plt.axvline(np.mean(uniform_bookings_econ_list), color='deepskyblue', linestyle='dashed', linewidth=1, label=f"Uniform Mean: {np.mean(uniform_bookings_econ_list):.2f}")
    plt.title('Monte Carlo Simulation of Total Economy Bookings')
    plt.xlabel('Total Economy Bookings')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Histogram of Total Bookings - Business
    plt.figure(figsize=(14, 6))
    plt.hist(dynamic_bookings_bus_list, bins=50, alpha=0.7, label='Dynamic Pricing', color='red')
    plt.hist(uniform_bookings_bus_list, bins=50, alpha=0.7, label='Uniform Pricing', color='salmon')
    plt.axvline(np.mean(dynamic_bookings_bus_list), color='darkred', linestyle='dashed', linewidth=1, label=f"Dynamic Mean: {np.mean(dynamic_bookings_bus_list):.2f}")
    plt.axvline(np.mean(uniform_bookings_bus_list), color='lightcoral', linestyle='dashed', linewidth=1, label=f"Uniform Mean: {np.mean(uniform_bookings_bus_list):.2f}")
    plt.title('Monte Carlo Simulation of Total Business Bookings')
    plt.xlabel('Total Business Bookings')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Average Bookings Over Time with Confidence Intervals - Economy
    all_bookings_economy_dynamic = np.array(dynamic_bookings_economy_daily_all)
    all_bookings_economy_uniform = np.array(uniform_bookings_economy_daily_all)

    mean_bookings_econ_dynamic = np.mean(all_bookings_economy_dynamic, axis=0)
    lower_bookings_econ_dynamic = np.percentile(all_bookings_economy_dynamic, 2.5, axis=0)
    upper_bookings_econ_dynamic = np.percentile(all_bookings_economy_dynamic, 97.5, axis=0)

    mean_bookings_econ_uniform = np.mean(all_bookings_economy_uniform, axis=0)
    lower_bookings_econ_uniform = np.percentile(all_bookings_economy_uniform, 2.5, axis=0)
    upper_bookings_econ_uniform = np.percentile(all_bookings_economy_uniform, 97.5, axis=0)

    days = np.arange(x_start, 0, -1)

    plt.figure(figsize=(14, 7))
    plt.plot(days, mean_bookings_econ_dynamic, label='Dynamic Pricing', color='blue')
    plt.fill_between(days, lower_bookings_econ_dynamic, upper_bookings_econ_dynamic, color='blue', alpha=0.2)
    plt.plot(days, mean_bookings_econ_uniform, label='Uniform Pricing', color='skyblue')
    plt.fill_between(days, lower_bookings_econ_uniform, upper_bookings_econ_uniform, color='skyblue', alpha=0.2)
    plt.xlabel('Days Before Departure')
    plt.ylabel('Number of Economy Bookings')
    plt.title('Average Daily Economy Bookings with 95% Confidence Intervals')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Average Bookings Over Time with Confidence Intervals - Business
    all_bookings_business_dynamic = np.array(dynamic_bookings_business_daily_all)
    all_bookings_business_uniform = np.array(uniform_bookings_business_daily_all)

    mean_bookings_bus_dynamic = np.mean(all_bookings_business_dynamic, axis=0)
    lower_bookings_bus_dynamic = np.percentile(all_bookings_business_dynamic, 2.5, axis=0)
    upper_bookings_bus_dynamic = np.percentile(all_bookings_business_dynamic, 97.5, axis=0)

    mean_bookings_bus_uniform = np.mean(all_bookings_business_uniform, axis=0)
    lower_bookings_bus_uniform = np.percentile(all_bookings_business_uniform, 2.5, axis=0)
    upper_bookings_bus_uniform = np.percentile(all_bookings_business_uniform, 97.5, axis=0)

    plt.figure(figsize=(14, 7))
    plt.plot(days, mean_bookings_bus_dynamic, label='Dynamic Pricing', color='red')
    plt.fill_between(days, lower_bookings_bus_dynamic, upper_bookings_bus_dynamic, color='red', alpha=0.2)
    plt.plot(days, mean_bookings_bus_uniform, label='Uniform Pricing', color='salmon')
    plt.fill_between(days, lower_bookings_bus_uniform, upper_bookings_bus_uniform, color='salmon', alpha=0.2)
    plt.xlabel('Days Before Departure')
    plt.ylabel('Number of Business Bookings')
    plt.title('Average Daily Business Bookings with 95% Confidence Intervals')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Average Prices Over Time
    all_prices_economy_dynamic = np.array(dynamic_prices_economy_daily_all)
    all_prices_business_dynamic = np.array(dynamic_prices_business_daily_all)

    all_prices_economy_uniform = np.array(uniform_prices_economy_daily_all)
    all_prices_business_uniform = np.array(uniform_prices_business_daily_all)

    mean_prices_econ_dynamic = np.mean(all_prices_economy_dynamic, axis=0)
    mean_prices_econ_uniform = np.mean(all_prices_economy_uniform, axis=0)

    mean_prices_bus_dynamic = np.mean(all_prices_business_dynamic, axis=0)
    mean_prices_bus_uniform = np.mean(all_prices_business_uniform, axis=0)

    plt.figure(figsize=(14, 7))
    plt.plot(days, mean_prices_econ_dynamic, label='Dynamic Pricing - Economy', color='blue')
    plt.plot(days, mean_prices_econ_uniform, label='Uniform Pricing - Economy', color='skyblue')
    plt.plot(days, mean_prices_bus_dynamic, label='Dynamic Pricing - Business', color='red')
    plt.plot(days, mean_prices_bus_uniform, label='Uniform Pricing - Business', color='salmon')
    plt.xlabel('Days Before Departure')
    plt.ylabel('Price ($)')
    plt.title('Average Daily Prices for Economy and Business Classes')
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
    num_simulations = 4000  # Adjust as needed
    
    # Run Monte Carlo simulations
    results = run_monte_carlo_simulations(
        num_simulations=num_simulations,
        params_economy=base_params_economy,
        params_business=base_params_business,
        capacity_economy=capacity_economy,
        capacity_business=capacity_business,
        x_start=x_start
    )
    
    (
        dynamic_revenues,
        dynamic_bookings_econ_list,
        dynamic_bookings_bus_list,
        dynamic_bookings_economy_daily_all,
        dynamic_bookings_business_daily_all,
        dynamic_prices_economy_daily_all,
        dynamic_prices_business_daily_all,
        uniform_revenues,
        uniform_bookings_econ_list,
        uniform_bookings_bus_list,
        uniform_bookings_economy_daily_all,
        uniform_bookings_business_daily_all,
        uniform_prices_economy_daily_all,
        uniform_prices_business_daily_all
    ) = results
    
    # Plot the Monte Carlo results
    plot_monte_carlo_results(
        dynamic_revenues,
        dynamic_bookings_econ_list,
        dynamic_bookings_bus_list,
        dynamic_bookings_economy_daily_all,
        dynamic_bookings_business_daily_all,
        dynamic_prices_economy_daily_all,
        dynamic_prices_business_daily_all,
        uniform_revenues,
        uniform_bookings_econ_list,
        uniform_bookings_bus_list,
        uniform_bookings_economy_daily_all,
        uniform_bookings_business_daily_all,
        uniform_prices_economy_daily_all,
        uniform_prices_business_daily_all,
        x_start,
        num_simulations
    )
    
    # Summary Statistics
    print(f"Total Simulations Run: {num_simulations}\n")
    
    print("=== Dynamic Pricing ===")
    print(f"Average Total Revenue: ${np.mean(dynamic_revenues):.2f}")
    print(f"Revenue 95% Confidence Interval: (${np.percentile(dynamic_revenues, 2.5):.2f}, ${np.percentile(dynamic_revenues, 97.5):.2f})")
    print(f"Average Total Economy Bookings: {np.mean(dynamic_bookings_econ_list):.2f}")
    print(f"Economy Bookings 95% Confidence Interval: ({np.percentile(dynamic_bookings_econ_list, 2.5):.0f}, {np.percentile(dynamic_bookings_econ_list, 97.5):.0f})")
    print(f"Average Total Business Bookings: {np.mean(dynamic_bookings_bus_list):.2f}")
    print(f"Business Bookings 95% Confidence Interval: ({np.percentile(dynamic_bookings_bus_list, 2.5):.0f}, {np.percentile(dynamic_bookings_bus_list, 97.5):.0f})\n")
    
    print("=== Uniform Pricing ===")
    print(f"Average Total Revenue: ${np.mean(uniform_revenues):.2f}")
    print(f"Revenue 95% Confidence Interval: (${np.percentile(uniform_revenues, 2.5):.2f}, ${np.percentile(uniform_revenues, 97.5):.2f})")
    print(f"Average Total Economy Bookings: {np.mean(uniform_bookings_econ_list):.2f}")
    print(f"Economy Bookings 95% Confidence Interval: ({np.percentile(uniform_bookings_econ_list, 2.5):.0f}, {np.percentile(uniform_bookings_econ_list, 97.5):.0f})")
    print(f"Average Total Business Bookings: {np.mean(uniform_bookings_bus_list):.2f}")
    print(f"Business Bookings 95% Confidence Interval: ({np.percentile(uniform_bookings_bus_list, 2.5):.0f}, {np.percentile(uniform_bookings_bus_list, 97.5):.0f})")

if __name__ == "__main__":
    main()
