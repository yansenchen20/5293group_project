import numpy as np

def normalize_iris(image, inner_circle, outer_circle, output_size=(64, 512)):
    (x_p, y_p, r_p) = inner_circle
    (x_i, y_i, r_i) = outer_circle
    M, N = output_size
    normalized_iris = np.zeros(output_size, dtype=np.uint8)

    # Calculate distance and angle difference between pupil and iris centers
    d1 = np.sqrt((x_p - x_i)**2 + (y_p - y_i)**2)
    angle_diff = np.arctan2(y_i - y_p, x_i - x_p)

    for X in range(N):
        theta = 2 * np.pi * X / N  # Angular position in the polar transformation

        # Calculate dynamic radius to account for non-concentric circles
        long_radius = (2 * d1 * np.cos(theta) + np.sqrt((2 * d1 * np.cos(theta))**2 - 4 * (d1**2 - r_i**2))) / 2
        for Y in range(M):
            radius = r_p + ((long_radius - r_p) * Y / M)  # Interpolate between inner and outer radii

            # Convert polar to Cartesian coordinates
            x_inner = x_p + r_p * np.cos(theta)
            y_inner = y_p + r_p * np.sin(theta)
            x_outer = x_p + long_radius * np.cos(theta)
            y_outer = y_p + long_radius * np.sin(theta)

            # Interpolate between the inner and outer boundaries
            x = int(x_inner + (x_outer - x_inner) * Y / M)
            y = int(y_inner + (y_outer - y_inner) * Y / M)

            # Assign pixel value if within image boundaries
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                normalized_iris[Y, X] = image[y, x]

    return normalized_iris
