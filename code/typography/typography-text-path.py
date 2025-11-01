# ----------------------------------------------------------------------------
# Title:   Scientific Visualisation - Python & Matplotlib
# Author:  Nicolas P. Rougier
# License: BSD
# Modified: 2025 by katoy & ChatGPT for macOS + Matplotlib 3.9+
# ----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties


# ----------------------------------------------------------------------------
# Interpolate helper
# ----------------------------------------------------------------------------
def interpolate(X, Y, T):
    """Interpolate X,Y coordinates along the parametric distance T."""
    if len(X) < 2 or len(Y) < 2:
        return X, Y, 0
    dR = (np.diff(X) ** 2 + np.diff(Y) ** 2) ** 0.5
    R = np.zeros_like(X)
    R[1:] = np.cumsum(dR)
    return np.interp(T, R, X), np.interp(T, R, Y), R[-1]


# ----------------------------------------------------------------------------
# Draw text along a contour line
# ----------------------------------------------------------------------------
def contour(ax, X, Y, text, offset=0):
    """Render given text along a curved line defined by X,Y."""
    if len(X) < 2 or len(Y) < 2:
        return

    # Interpolate text along curve
    # X0,Y0 for position + X1,Y1 for normal vectors
    path = TextPath(
        (0, -0.75),
        text,
        prop=FontProperties(size=2, family="sans-serif", weight="bold"),
    )

    # vertices must be writable (copy to avoid read-only view)
    V = path.vertices.copy()

    X0, Y0, D = interpolate(X, Y, offset + V[:, 0])
    X1, Y1, _ = interpolate(X, Y, offset + V[:, 0] + 0.1)

    # Here we interpolate the original path to get the "remainder"
    # (path minus text)
    Xr, Yr, _ = interpolate(X, Y, np.linspace(V[:, 0].max() + 1, D - 1, 200))
    ax.plot(
        Xr,
        Yr,
        color="black",
        linewidth=0.5,
        markersize=1,
        marker="o",
        markevery=[0, -1],
    )

    # Transform text vertices
    dX, dY = X1 - X0, Y1 - Y0
    norm = np.sqrt(dX**2 + dY**2)
    if np.any(norm == 0):
        return
    dX, dY = dX / norm, dY / norm
    X0 += -V[:, 1] * dY
    Y0 += +V[:, 1] * dX
    V[:, 0], V[:, 1] = X0, Y0

    # Faint outline
    patch = PathPatch(
        path,
        facecolor="white",
        zorder=10,
        alpha=0.25,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.add_artist(patch)

    # Actual text
    patch = PathPatch(
        path, facecolor="black", zorder=30, edgecolor="black", linewidth=0.0
    )
    ax.add_artist(patch)


# ----------------------------------------------------------------------------
# Some data
# ----------------------------------------------------------------------------
def f(x, y):
    """Sample bivariate function for the contour demo."""
    return (1 - x / 2 + x**5 + y**3) * np.exp(-(x**2) - y**2)


n = 100
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y)
Z = 0.5 * f(X, Y)


# ----------------------------------------------------------------------------
# Plot setup
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 5), dpi=100)
levels = 10

# Regular contour with straight labels
ax1 = fig.add_subplot(1, 2, 1, aspect=1, xticks=[], yticks=[])
CF = ax1.contourf(Z, origin="lower", levels=levels)
CS = ax1.contour(Z, origin="lower", levels=levels, colors="black", linewidths=0.5)
ax1.clabel(CS, CS.levels)

# Regular contour with curved labels
# ! aspect=1 is critical here, else text path would be deformed
ax2 = fig.add_subplot(1, 2, 2, aspect=1, xticks=[], yticks=[])
CF = ax2.contourf(Z, origin="lower", levels=levels)
CS = ax2.contour(
    Z, origin="lower", levels=levels, alpha=0, colors="black", linewidths=0.5
)

# Matplotlib 3.9+ : get_paths() returns a flat list of Path objects
paths = CS.get_paths()
levels_iter = np.linspace(Z.min(), Z.max(), len(paths))  # pseudo-level mapping

for level, path in zip(levels_iter, paths):
    V = np.array(path.vertices)
    if V.size == 0:
        continue
    text = f"{level:.3f}"
    if abs(level) < 1e-6:
        text = "  DO NOT CROSS  •••" * 8
    contour(ax2, V[:, 0], V[:, 1], text)

plt.tight_layout()
plt.show()
