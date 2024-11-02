import cv2
import numpy as np

def enhance_iris(image):

    '''
    1.Lighting Adjustment
    '''
    block_size = 16  # Block size for coarse background estimation
    height, width = image.shape

    # Coarse Background Estimation (Average intensity for each 16x16 block)
    small_blocks = image.reshape(height // block_size,
                                 block_size,
                                 width // block_size,
                                 block_size).mean(axis=(1, 3))

    # Expand the background estimation to match the original image size using bicubic interpolation
    background_estimation = cv2.resize(small_blocks, (width, height), interpolation=cv2.INTER_CUBIC)

    # Subtract estimated background from the original image
    lighting_corrected = cv2.subtract(image, background_estimation.astype(np.uint8))

    '''
    2. Local Histogram Equalization
    '''
    # Initialize matrix for storing the enhanced image
    enhanced_image = np.zeros_like(lighting_corrected)
    local_block_size = 32  # Block size for local histogram equalization

    # Apply histogram equalization in each 32x32 region
    for i in range(0, height, local_block_size): # through rows
        for j in range(0, width, local_block_size): # through columns
            # Define the block region, handling edges where block might be smaller
            region = lighting_corrected[i : min(i + local_block_size, height),
                                        j : min(j + local_block_size, width)]
            # Equalize the region
            equalized_region = cv2.equalizeHist(region)
            # Place the equalized region back into the enhanced image
            enhanced_image[i : i + equalized_region.shape[0],
                           j : j + equalized_region.shape[1]] = equalized_region

    return enhanced_image
