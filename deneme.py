import numpy as np
import matplotlib.pyplot as plt


def gaussian_radiation_pattern(sigma_el=15.0, sigma_az=15.0):
    elevation = np.linspace(-90, 90, 180)
    azimuth = np.linspace(-180, 180, 360)
    el_grid, az_grid = np.meshgrid(elevation, azimuth, indexing="ij")

    norm = 1.0 / (2.0 * np.pi * sigma_el * sigma_az)
    pdf = norm * np.exp(
        -0.5 * ((el_grid / sigma_el) ** 2 + (az_grid / sigma_az) ** 2)
    )

    matrix = np.column_stack((el_grid.ravel(), az_grid.ravel(), pdf.ravel()))
    return elevation, azimuth, pdf, matrix


if __name__ == "__main__":
    elevation, azimuth, pdf, antenna_radiation_pattern = gaussian_radiation_pattern()

    plt.figure(figsize=(7, 4))
    mesh = plt.pcolormesh(
        azimuth,
        elevation,
        pdf,
        cmap="Greys",
        shading="auto",
    )
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Elevation (deg)")
    plt.title("Antenna Radiation Pattern (Gaussian)")
    plt.colorbar(mesh, label="Probability Density")
    plt.tight_layout()
    plt.show()