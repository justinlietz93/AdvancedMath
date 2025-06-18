import numpy as np
import matplotlib.pyplot as plt
from src.tda import compute_persistent_homology, calculate_tda_metrics
from ripser import plot_dgms

def example_two_clusters():
    """
    An example demonstrating TDA on a point cloud with two distinct clusters.
    """
    print("--- Example: Two Clusters ---")
    # Generate data: two clusters of points
    points = np.vstack([
        np.random.randn(20, 2) + [2, 2],
        np.random.randn(20, 2) + [-2, -2]
    ])

    # Compute persistent homology
    result = compute_persistent_homology(points, max_dim=1)
    diagrams = result['dgms']

    # Calculate TDA metrics
    metrics = calculate_tda_metrics(diagrams)
    print(f"TDA Metrics: {metrics}")

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(points[:, 0], points[:, 1], s=10)
    plt.title("Point Cloud: Two Clusters")
    
    plt.subplot(1, 2, 2)
    plot_dgms(diagrams, show=False)
    plt.title("Persistence Diagram")
    
    plt.tight_layout()
    plt.savefig("tda_example_two_clusters.png")
    plt.show()

def example_circle():
    """
    An example demonstrating TDA on a point cloud forming a circle,
    which should have a prominent 1-dimensional feature (a loop).
    """
    print("\n--- Example: Circle ---")
    # Generate data: points on a circle
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    points = np.array([np.cos(t), np.sin(t)]).T
    points += 0.1 * np.random.randn(*points.shape)

    # Compute persistent homology
    result = compute_persistent_homology(points, max_dim=1)
    diagrams = result['dgms']

    # Calculate TDA metrics
    metrics = calculate_tda_metrics(diagrams)
    print(f"TDA Metrics: {metrics}")

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(points[:, 0], points[:, 1], s=10)
    plt.title("Point Cloud: Circle")
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plot_dgms(diagrams, show=False)
    plt.title("Persistence Diagram")

    plt.tight_layout()
    plt.savefig("tda_example_circle.png")
    plt.show()

if __name__ == '__main__':
    example_two_clusters()
    example_circle()
