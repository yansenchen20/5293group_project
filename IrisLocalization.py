import cv2
import numpy as np

def localize_iris(image):
    # Step 1: Estimate approximate center of the pupil using projections on a sub-image
    subImage = image[60:240, 100:220]
    vertical_projection = np.sum(subImage, axis=1)
    horizontal_projection = np.sum(subImage, axis=0)

    y_center = np.argmin(vertical_projection) + 60
    x_center = np.argmin(horizontal_projection) + 100

    # Step 2: Use a 120x120 region centered at the estimated center for thresholding and mask creation
    height, width = image.shape
    x_min = max(0, x_center - 60)
    x_max = min(width, x_center + 60)
    y_min = max(0, y_center - 60)
    y_max = min(height, y_center + 60)
    region120 = image[y_min:y_max, x_min:x_max]

    ret, binary_roi = cv2.threshold(region120, 64, 65, cv2.THRESH_BINARY)
    mask = np.where(binary_roi > 0, 1, 0)

    # Refine the pupil center using mask projections
    vertical_sum = mask.sum(axis=0)
    horizontal_sum = mask.sum(axis=1)
    min_y = np.argmin(horizontal_sum)
    min_x = np.argmin(vertical_sum)

    radius1 = (120 - horizontal_sum[min_y]) / 2
    radius2 = (120 - vertical_sum[min_x]) / 2
    pupil_radius = int((radius1 + radius2) / 2)

    refined_y_center = min_y + y_center - 60
    refined_x_center = min_x + x_center - 60

    # Step 3: Define a larger region for Hough circle detection on the iris boundary
    region_pupil = image[max(0, refined_y_center - 60):min(height, refined_y_center + 60),
                         max(0, refined_x_center - 60):min(width, refined_x_center + 60)]

    region_iris = image[max(0, refined_y_center - 120):min(height, refined_y_center + 110),
                        max(0, refined_x_center - 135):min(width, refined_x_center + 135)]

    # Step 4: Hough circle detection for pupil
    for loop in range(1, 5):
        pupil_circles = cv2.HoughCircles(region_pupil, cv2.HOUGH_GRADIENT, 1, 250,
                                         param1=50, param2=10, minRadius=(pupil_radius - loop), maxRadius=(pupil_radius + loop))
        if pupil_circles is not None:
            pupil_circles = np.round(pupil_circles[0, :]).astype(int)
            inner_circle = [pupil_circles[0][0] + refined_x_center - 60,
                            pupil_circles[0][1] + refined_y_center - 60,
                            pupil_circles[0][2]]
            break

    # Step 5: Hough circle detection for iris
    iris_circles = cv2.HoughCircles(region_iris, cv2.HOUGH_GRADIENT, 1, 250,
                                    param1=30, param2=10, minRadius=98, maxRadius=118)
    if iris_circles is not None:
        iris_circles = np.round(iris_circles[0, :]).astype(int)
        outer_circle = [iris_circles[0][0] + refined_x_center - 135,
                        iris_circles[0][1] + refined_y_center - 120,
                        iris_circles[0][2]]
    else:
        outer_circle = None

    # Step 6: Return detected circles or None if either is not found
    if inner_circle is not None and outer_circle is not None:
        return inner_circle, outer_circle
    else:
        return None
