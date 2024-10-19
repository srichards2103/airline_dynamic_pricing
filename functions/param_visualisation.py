import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import seaborn as sns

# Set the style using seaborn (no need to apply matplotlib style separately)
sns.set(style="whitegrid")
sns.set_palette("husl")

def demand(x, g, d, h, a, b):
    """Updated demand function that incorporates probability."""
    return (g * x + d) * np.exp(-h * x) * np.exp(-100 * (a + b * x))

def probability(y, a, b, x):
    """Probability function (unchanged)."""
    return np.exp(-y * (a + b * x))

def plot_demand(ax, x, params, label, color):
    """Updated plot_demand function."""
    y = demand(x, params['g'], params['d'], params['h'], params['a'], params['b'])
    ax.plot(x, y, label=label, color=color, linewidth=3)
    ax.set_xlabel('Days Before Departure', fontsize=12)
    ax.set_ylabel('Demand', fontsize=12)
    ax.set_title('Demand Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

def plot_probability(ax, x, params, label, color):
    """Updated plot_probability function."""
    y_prices = np.linspace(50, 800, 500)
    x_avg = np.mean(x)
    p = probability(y_prices, params['a'], params['b'], x_avg)
    ax.plot(y_prices, p, label=label, color=color, linewidth=3)
    ax.set_xlabel('Price ($)', fontsize=12)
    ax.set_ylabel('Probability of Purchase', fontsize=12)
    ax.set_title('Probability Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

def interactive_visualization():
    """Updated interactive visualization function."""
    params = {
        'Business': {'g': 5, 'd': 10, 'h': 0.1, 'a': 0.0067, 'b': 0.00291},
        'Economy': {'g': 10, 'd': 30, 'h': 0.117, 'a': 0.0067, 'b': 0.00291}
    }

    x = np.arange(30, -1, -1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(left=0.1, bottom=0.35)

    plot_demand(ax1, x, params['Business'], label='Business', color='#FF9999')
    plot_demand(ax1, x, params['Economy'], label='Economy', color='#66B2FF')
    plot_probability(ax2, x, params['Business'], label='Business', color='#FF9999')
    plot_probability(ax2, x, params['Economy'], label='Economy', color='#66B2FF')

    slider_axes = [
        plt.axes([0.1, 0.25 - i*0.05, 0.3, 0.03]) for i in range(5)
    ] + [
        plt.axes([0.55, 0.25 - i*0.05, 0.3, 0.03]) for i in range(5)
    ]

    sliders = {}
    for i, (param, (min_val, max_val)) in enumerate([
        ('g_B', (0.5, 25)), ('d_B', (10, 100)), ('h_B', (0.01, 0.2)),
        ('a_B', (0.001, 0.01)), ('b_B', (0.001, 0.008)),
        ('g_E', (0.5, 25)), ('d_E', (5, 100)), ('h_E', (0.01, 0.2)),
        ('a_E', (0.001, 0.01)), ('b_E', (0.001, 0.008))
    ]):
        # Extract the class ('Business' or 'Economy') from the parameter name
        class_type = 'Business' if '_B' in param else 'Economy'
        # Extract the key (the first character of the param) and access the correct value in params
        param_key = param[0].lower()
        sliders[param] = Slider(slider_axes[i], param, min_val, max_val,
                                valinit=params[class_type][param_key])

    def update(val):
        for param, slider in sliders.items():
            class_type = 'Business' if '_B' in param else 'Economy'
            param_key = param[0].lower()  # Extract the correct key for updating
            params[class_type][param_key] = slider.val

        ax1.clear()
        ax2.clear()
        plot_demand(ax1, x, params['Business'], label='Business', color='#FF9999')
        plot_demand(ax1, x, params['Economy'], label='Economy', color='#66B2FF')
        plot_probability(ax2, x, params['Business'], label='Business', color='#FF9999')
        plot_probability(ax2, x, params['Economy'], label='Economy', color='#66B2FF')
        fig.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(update)

    reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(reset_ax, 'Reset', color='#CCCCCC', hovercolor='#AAAAAA')

    def reset(event):
        for slider in sliders.values():
            slider.reset()

    reset_button.on_clicked(reset)

    plt.suptitle('Interactive Demand and Probability Visualization', fontsize=16, fontweight='bold')
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
        y = demand(x, g, 50, 0.05, 0.0067, 0.00291)
        axes1[0].plot(x, y, label=f'g_B={g}')
    axes1[0].set_xlabel('Days Before Departure')
    axes1[0].set_ylabel('Demand (Business)')
    axes1[0].set_title('Business Class Demand - Varying g_B')
    axes1[0].legend()

    # Plot Economy Demand with varying 'g'
    for g in parameter_variations['Economy']['g']:
        y = demand(x, g, 150, 0.03, 0.0067, 0.00291)
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
