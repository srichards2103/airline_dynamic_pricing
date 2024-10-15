import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider, Button

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid")

# Define the demand and probability functions
def demand(x, g, d, h):
    """
    Demand function.

    Parameters:
        x (array): Days before departure.
        g (float): Demand slope.
        d (float): Demand intercept.
        h (float): Demand decay rate.

    Returns:
        array: Demand values.
    """
    return (g * x + d) * np.exp(-h * x)

def probability(y, a, b, x):
    """
    Probability function.

    Parameters:
        y (array): Price.
        a (float): Probability slope.
        b (float): Probability decay rate.
        x (array): Days before departure.

    Returns:
        array: Probability values.
    """
    return np.exp(-y * (a + b * x))

def plot_demand(ax, x, g, d, h, label, color):
    """
    Plots the demand curve.

    Parameters:
        ax (matplotlib.axes.Axes): Axis to plot on.
        x (array): Days before departure.
        g (float): Demand slope.
        d (float): Demand intercept.
        h (float): Demand decay rate.
        label (str): Label for the curve.
        color (str): Color for the curve.
    """
    y = demand(x, g, d, h)
    ax.plot(x, y, label=label, color=color)
    ax.set_xlabel('Days Before Departure')
    ax.set_ylabel('Demand')
    ax.set_title('Demand Curves')
    ax.legend()

def plot_probability(ax, x, a, b, label, color):
    """
    Plots the probability curve.

    Parameters:
        ax (matplotlib.axes.Axes): Axis to plot on.
        x (array): Days before departure.
        a (float): Probability slope.
        b (float): Probability decay rate.
        label (str): Label for the curve.
        color (str): Color for the curve.
    """
    # Define a range of prices for visualization
    y_prices = np.linspace(50, 800, 500)
    # For visualization, fix days before departure or average
    # Here, we take an average scenario
    x_avg = np.mean(x)
    p = probability(y_prices, a, b, x_avg)
    ax.plot(y_prices, p, label=label, color=color)
    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Probability of Purchase')
    ax.set_title('Probability Curves')
    ax.legend()

def interactive_visualization():
    """
    Creates an interactive visualization to explore how parameters affect demand and probability curves.
    """
    # Initial Parameters for Business and Economy Classes
    params = {
        'Business': {
            'g': 5,
            'd': 10,
            'h': 0.1,
            'a': 0.0067,
            'b': 0.00291
        },
        'Economy': {
            'g': 10,
            'd': 30,
            'h': 0.117,
            'a': 0.0067,
            'b': 0.00291
        }
    }

    # Days before departure
    x = np.arange(30, -1, -1)  # 30 to 0

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(left=0.1, bottom=0.35)

    # Plot initial demand curves
    plot_demand(ax1, x, params['Business']['g'], params['Business']['d'], params['Business']['h'],
               label='Business', color='blue')
    plot_demand(ax1, x, params['Economy']['g'], params['Economy']['d'], params['Economy']['h'],
               label='Economy', color='green')

    # Plot initial probability curves
    plot_probability(ax2, x, params['Business']['a'], params['Business']['b'],
                    label='Business', color='blue')
    plot_probability(ax2, x, params['Economy']['a'], params['Economy']['b'],
                    label='Economy', color='green')

    # Define Slider axes
    axcolor = 'lightgoldenrodyellow'
    slider_width = 0.3
    slider_height = 0.02
    spacing = 0.05

    # Create sliders for Business class parameters
    ax_gB = plt.axes([0.1, 0.25, slider_width, slider_height], facecolor=axcolor)
    ax_dB = plt.axes([0.1, 0.20, slider_width, slider_height], facecolor=axcolor)
    ax_hB = plt.axes([0.1, 0.15, slider_width, slider_height], facecolor=axcolor)
    ax_aB = plt.axes([0.55, 0.25, slider_width, slider_height], facecolor=axcolor)
    ax_bB = plt.axes([0.55, 0.20, slider_width, slider_height], facecolor=axcolor)

    # Create sliders for Economy class parameters
    ax_gE = plt.axes([0.1, 0.10, slider_width, slider_height], facecolor=axcolor)
    ax_dE = plt.axes([0.1, 0.05, slider_width, slider_height], facecolor=axcolor)
    ax_hE = plt.axes([0.55, 0.10, slider_width, slider_height], facecolor=axcolor)
    ax_aE = plt.axes([0.55, 0.05, slider_width, slider_height], facecolor=axcolor)
    ax_bE = plt.axes([0.55, 0.00, slider_width, slider_height], facecolor=axcolor)

    # Initialize sliders
    slider_gB = Slider(ax_gB, 'g_B', 0.5, 25, valinit=params['Business']['g'])
    slider_dB = Slider(ax_dB, 'd_B', 10, 100, valinit=params['Business']['d'])
    slider_hB = Slider(ax_hB, 'h_B', 0.01, 0.2, valinit=params['Business']['h'])
    slider_aB = Slider(ax_aB, 'a_B', 0.001, 0.01, valinit=params['Business']['a'])
    slider_bB = Slider(ax_bB, 'b_B', 0.001, 0.008, valinit=params['Business']['b'])

    slider_gE = Slider(ax_gE, 'g_E', 0.5,25, valinit=params['Economy']['g'])
    slider_dE = Slider(ax_dE, 'd_E', 5, 100, valinit=params['Economy']['d'])
    slider_hE = Slider(ax_hE, 'h_E', 0.01, 0.2, valinit=params['Economy']['h'])
    slider_aE = Slider(ax_aE, 'a_E', 0.001, 0.01, valinit=params['Economy']['a'])
    slider_bE = Slider(ax_bE, 'b_E', 0.001, 0.008, valinit=params['Economy']['b'])

    # Update function
    def update(val):
        # Update parameters based on slider values
        params['Business']['g'] = slider_gB.val
        params['Business']['d'] = slider_dB.val
        params['Business']['h'] = slider_hB.val
        params['Business']['a'] = slider_aB.val
        params['Business']['b'] = slider_bB.val

        params['Economy']['g'] = slider_gE.val
        params['Economy']['d'] = slider_dE.val
        params['Economy']['h'] = slider_hE.val
        params['Economy']['a'] = slider_aE.val
        params['Economy']['b'] = slider_bE.val

        # Clear previous plots
        ax1.cla()
        ax2.cla()

        # Re-plot demand curves
        plot_demand(ax1, x, params['Business']['g'], params['Business']['d'], params['Business']['h'],
                   label='Business', color='blue')
        plot_demand(ax1, x, params['Economy']['g'], params['Economy']['d'], params['Economy']['h'],
                   label='Economy', color='green')

        # Re-plot probability curves
        plot_probability(ax2, x, params['Business']['a'], params['Business']['b'],
                        label='Business', color='blue')
        plot_probability(ax2, x, params['Economy']['a'], params['Economy']['b'],
                        label='Economy', color='green')

        fig.canvas.draw_idle()

    # Register the update function with each slider
    slider_gB.on_changed(update)
    slider_dB.on_changed(update)
    slider_hB.on_changed(update)
    slider_aB.on_changed(update)
    slider_bB.on_changed(update)
    slider_gE.on_changed(update)
    slider_dE.on_changed(update)
    slider_hE.on_changed(update)
    slider_aE.on_changed(update)
    slider_bE.on_changed(update)

    # Add a Reset button
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        slider_gB.reset()
        slider_dB.reset()
        slider_hB.reset()
        slider_aB.reset()
        slider_bB.reset()
        slider_gE.reset()
        slider_dE.reset()
        slider_hE.reset()
        slider_aE.reset()
        slider_bE.reset()

    button.on_clicked(reset)

    plt.show()

def static_visualization():
    """
    Creates static plots to visualize how different parameters affect demand and probability curves.
    """
    # Define a set of parameter variations for demonstration
    parameter_variations = {
        'Business': {
            'g': [1.5, 2.0, 2.5],
            'd': [40, 50, 60],
            'h': [0.04, 0.05, 0.06]
        },
        'Economy': {
            'g': [4.5, 5.0, 5.5],
            'd': [140, 150, 160],
            'h': [0.025, 0.03, 0.035]
        }
    }

    # Days before departure
    x = np.arange(30, -1, -1)  # 30 to 0

    # Create subplots for Demand
    fig1, axes1 = plt.subplots(2, 1, figsize=(10, 12))
    fig1.suptitle('Effect of Parameters on Demand Curves', fontsize=16)

    # Plot Business Demand with varying 'g'
    for g in parameter_variations['Business']['g']:
        y = demand(x, g, 50, 0.05)
        axes1[0].plot(x, y, label=f'g_B={g}')
    axes1[0].set_xlabel('Days Before Departure')
    axes1[0].set_ylabel('Demand (Business)')
    axes1[0].set_title('Business Class Demand - Varying g_B')
    axes1[0].legend()

    # Plot Economy Demand with varying 'g'
    for g in parameter_variations['Economy']['g']:
        y = demand(x, g, 150, 0.03)
        axes1[1].plot(x, y, label=f'g_E={g}')
    axes1[1].set_xlabel('Days Before Departure')
    axes1[1].set_ylabel('Demand (Economy)')
    axes1[1].set_title('Economy Class Demand - Varying g_E')
    axes1[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Create subplots for Probability
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 12))
    fig2.suptitle('Effect of Parameters on Probability Curves', fontsize=16)

    # Define a fixed day for probability visualization (e.g., 15 days before departure)
    x_fixed = 15

    # Plot Business Probability with varying 'a' and 'b'
    for a in [0.08, 0.1, 0.12]:
        for b in [0.004, 0.005, 0.006]:
            y_prices = np.linspace(100, 800, 500)
            p = probability(y_prices, a, b, x_fixed)
            axes2[0].plot(y_prices, p, label=f'a_B={a}, b_B={b}')
    axes2[0].set_xlabel('Price ($)')
    axes2[0].set_ylabel('Probability of Purchase (Business)')
    axes2[0].set_title('Business Class Probability - Varying a_B and b_B')
    axes2[0].legend()

    # Plot Economy Probability with varying 'a' and 'b'
    for a in [0.04, 0.05, 0.06]:
        for b in [0.0018, 0.002, 0.0022]:
            y_prices = np.linspace(50, 500, 500)
            p = probability(y_prices, a, b, x_fixed)
            axes2[1].plot(y_prices, p, label=f'a_E={a}, b_E={b}')
    axes2[1].set_xlabel('Price ($)')
    axes2[1].set_ylabel('Probability of Purchase (Economy)')
    axes2[1].set_title('Economy Class Probability - Varying a_E and b_E')
    axes2[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    """
    Main function to run visualizations.
    """
    choice = input("Choose visualization type:\n1. Interactive Visualization\n2. Static Visualization\nEnter 1 or 2: ")
    if choice == '1':
        interactive_visualization()
    elif choice == '2':
        static_visualization()
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
