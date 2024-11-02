import numpy as np
import scipy.signal

# Extracting region of interest (top 48 rows as stated in the paper)
def get_roi(image):
    return image[:48, :]

# Define the Gabor-like spatial filter with cosine modulation (as stated in the paper)
# sigma values are also specified in the paper; we will incorporate them when executing the enhancement
def get_filter_kernel(sigma_x, sigma_y):
    kernel_size = 9
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    f = 1 / sigma_y  # Frequency

    for i in range(kernel_size):
        for j in range(kernel_size):
            x = j - center
            y = i - center
            gaussian = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(-((x**2 / sigma_x**2) + (y**2 / sigma_y**2)) / 2)
            modulation = np.cos(2 * np.pi * f * np.sqrt(x**2 + y**2))
            kernel[i, j] = gaussian * modulation

    return kernel

# Apply the spatial filter to the region of interest
def apply_spatial_filter(roi, sigma_x, sigma_y):
    kernel = get_filter_kernel(sigma_x, sigma_y)
    # Convolve the filter kernel with the region of interest
    filtered_image = scipy.signal.convolve2d(roi, kernel, mode='same')
    return filtered_image

# Extract the feature vector as described in the paper with explicit indexing
def get_feature_vector(image):
    roi = get_roi(image)

    # Apply two different filters with specified sigma values
    filtered_image1 = apply_spatial_filter(roi, sigma_x=3, sigma_y=1.5)
    filtered_image2 = apply_spatial_filter(roi, sigma_x=4.5, sigma_y=1.5)

    # Initialize the feature vector
    n_blocks_row = 48 // 8  # Number of 8x8 blocks in height
    n_blocks_col = 512 // 8  # Number of 8x8 blocks in width
    feature_vector = np.zeros(n_blocks_row * n_blocks_col * 4)  # 4 values per block (2 means, 2 deviations)

    index = 0
    for filtered_image in [filtered_image1, filtered_image2]:
        for row in range(n_blocks_row):
            for col in range(n_blocks_col):
                block = filtered_image[row*8:(row+1)*8, col*8:(col+1)*8]

                # Calculate mean and standard deviation for the block
                mean_value = np.mean(np.abs(block))
                std_dev = np.mean(np.abs(block - mean_value))

                # Assign values to feature vector
                feature_vector[index] = mean_value
                feature_vector[index + 1] = std_dev
                index += 2  # Move to the next pair in the feature vector

    return feature_vector
