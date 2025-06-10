import json
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import os

# Path to the directory containing JSON files
directory_path = r'C:\Users\YOURUSERNAME\AppData\Roaming\CustomHeadset\Distortion'

def load_points_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Extract the distortions from the JSON data and convert them to a NumPy array
    points = np.array(data['distortions']).reshape(-1, 2)
    return points


def smooth_points(points, inner_point_counts):
    x = points[:, 0]
    y = points[:, 1]

    # Ensure there are at least two control points for interpolation
    if len(x) < 2:
        raise ValueError("At least two control points are required.")

    cs = CubicSpline(x, y)

    # Generate more points between the existing ones for a smoother curve
    x_smoothed = np.linspace(x.min(), x.max(), len(x) * inner_point_counts)
    y_smoothed = cs(x_smoothed)

    return np.column_stack((x_smoothed, y_smoothed))


def calculate_integral(points):
    return np.cumsum(points[:, 1])


def calculate_derivative(points):
    return np.gradient(points[:, 1], points[:, 0])


# Get a list of all .json files in the directory
json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]

# Number of inner points per segment for smoothing
inner_point_counts = 20

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Initialize variables to store the minimum and maximum x-values
x_min = float('inf')
x_max = float('-inf')

for json_file in json_files:
    file_path = os.path.join(directory_path, json_file)

    try:
        points = load_points_from_json(file_path)

        if len(points) < 2:
            raise ValueError("At least two control points are required.")

        # Update the minimum and maximum x-values
        x_min = min(x_min, points[:, 0].min())
        x_max = max(x_max, points[:, 0].max())

        smoothed_points = smooth_points(points, inner_point_counts)

        # Calculate integral and derivative
        integral = calculate_integral(smoothed_points)
        derivative = calculate_derivative(smoothed_points)

        # Plot original points
        # ax.plot(points[:, 0], points[:, 1], 'ro', label=f'Original Points - {json_file}')

        # Plot smoothed curve
        # ax.plot(smoothed_points[:, 0], smoothed_points[:, 1], label=f'Smoothed Curve - {json_file}')

        # Plot integral
        # ax.plot(smoothed_points[:, 0], integral, label=f'Integral - {json_file}')

        # Plot derivative
        ax.plot(smoothed_points[:, 0], derivative, label=f'Derivative - {json_file}')
    except Exception as e:
        print(f"Error processing file {json_file}: {e}")

# Set labels and title
ax.set_title('Distortion Profiles Comparison')
ax.set_xlabel('Radius from center of output (%)')
ax.set_ylabel('Radius from center of input (degrees)')
ax.legend()
plt.grid(True)

# Ensure the x-axis covers the full range of data
ax.set_xlim(x_min, x_max)

plt.tight_layout()

# Show the plot
plt.show()
