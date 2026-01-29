import json

import matplotlib.pyplot as plt
import numpy as np

# Load the data from the JSON file
with open("/home/isokolov/qex/scripts/noise_evaluation_results_ksr3.json") as f:
    data = json.load(f)

# Extract the data
noise_levels = data["noise_levels"]
losses = data["losses"]
convergence = data["convergence"]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the losses vs noise levels on a log-log scale
plt.loglog(noise_levels, losses, "o-", linewidth=2, markersize=8)

# Add labels and title
plt.xlabel("Noise Level", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Loss vs Noise Level", fontsize=14)

# Add grid for better readability
plt.grid(True, which="both", ls="-", alpha=0.2)

# # Annotate convergence status
# for i, (x, y, conv) in enumerate(zip(noise_levels, losses, convergence)):
#     color = "red" if not conv else "green"
#     marker = "✗" if not conv else "✓"
#     plt.annotate(
#         f"{marker}",
#         xy=(x, y),
#         xytext=(5, 5),
#         textcoords="offset points",
#         color=color,
#         fontsize=12,
#         weight="bold",
#     )

# Add a legend explaining the markers
from matplotlib.lines import Line2D

legend_elements = []

# legend_elements = [
#     Line2D(
#         [0],
#         [0],
#         marker="x",
#         color="w",
#         markerfacecolor="red",
#         markersize=10,
#         label="Not Converged",
#     ),
#     Line2D(
#         [0],
#         [0],
#         marker="d",
#         color="w",
#         markerfacecolor="green",
#         markersize=10,
#         label="Converged",
#     ),
# ]

# Add fit through last 2 data points
if len(noise_levels) >= 2:
    # Get the last two points
    x_fit = noise_levels[-5:]
    y_fit = losses[-5:]

    # Calculate slope and intercept in log space (for power law fit)
    log_x = np.log10(x_fit)
    log_y = np.log10(y_fit)
    slope = (log_y[1] - log_y[0]) / (log_x[1] - log_x[0])

    # Create extended x range for the fit line - Fix to avoid log(0) issues
    min_noise = max(1e-10, min(noise_levels))  # Ensure minimum is positive
    x_extended = np.logspace(np.log10(0.1), np.log10(max(noise_levels)) * 1.1, 100)

    # Calculate the power law fit: y = A * x^slope
    A = y_fit[0] / (x_fit[0] ** slope)
    y_extended = A * (x_extended**slope)

    # Plot the fit line
    plt.loglog(
        x_extended,
        y_extended,
        "--",
        color="blue",
        linewidth=1.5,
        label=f"Fit: y ∝ x^{slope:.2f}",
    )

    # Add the fit to the legend
    legend_elements.append(
        Line2D(
            [0],
            [0],
            linestyle="--",
            color="blue",
            linewidth=1.5,
            label=f"Fit: y ∝ x^{slope:.2f}",
        ),
    )

plt.legend(handles=legend_elements, loc="best")

# Show the plot
plt.tight_layout()
plt.savefig("./qex/scripts/noise_evaluation_plot_ksr3.png", dpi=300)
plt.show()
