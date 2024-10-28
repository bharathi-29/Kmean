import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

# Generate synthetic data
X, y_true = make_blobs(n_samples=100, centers=4, cluster_std=0.60, random_state=0)
X = X[:, ::-1]  # Flip axes for better plotting

# Fit a Gaussian Mixture Model
gmm = GaussianMixture(n_components=4, random_state=42).fit(X)
labels = gmm.predict(X)

# Plot the data with color-coded labels
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap="viridis")
probs = gmm.predict_proba(X)
print(probs[:5].round(3))

# Adjust the size based on probabilities to emphasize differences
size = 50 * probs.max(1) ** 2
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=size)

# Function to draw an ellipse based on position and covariance
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))

# Function to plot Gaussian Mixture Model results
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap="viridis", zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    
    ax.axis("equal")
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor, ax=ax, edgecolor='black')

# Plot the GMM results
gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42)
plot_gmm(gmm, X)

plt.show()