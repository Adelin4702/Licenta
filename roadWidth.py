import cv2
import numpy as np

def get_max_road_width_y(image):
    if image is None:
        print("Error: Image not loaded properly.")
        return 0, None, None, None  # Modify to return more values

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection using Canny
    edges = cv2.Canny(gray, 50, 150)

    # Hough Line Transform to detect lane lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)

    if lines is None:
        print("No lane lines detected.")
        return 0, None, None, None  # Modify to return more values

    # Copy the original image to draw the lines on
    line_image = np.copy(image)

    # Extract leftmost and rightmost x-coordinates with corresponding y
    left_points = []
    right_points = []

    image_center = image.shape[1] // 2  # Midpoint of the image (assumed road center)

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Assign lines to left or right side
        if x1 < image_center and x2 < image_center:
            left_points.append((x1, y1))
            left_points.append((x2, y2))
        elif x1 > image_center and x2 > image_center:
            right_points.append((x1, y1))
            right_points.append((x2, y2))

        # Draw the line on the image (Hough line result)
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines for detected lanes

    if not left_points or not right_points:
        print("Could not detect both sides of the road.")
        return 0, None, None, None  # Modify to return more values

    # Get leftmost (min x) and rightmost (max x) points
    left_most = min(left_points, key=lambda p: p[0])  # (x, y) pair with min x
    right_most = max(right_points, key=lambda p: p[0])  # (x, y) pair with max x

    # Calculate maximum road width in pixels
    max_width = right_most[0] - left_most[0]

    print(f"Maximum Road Width: {max_width} pixels")
    print(f"Leftmost Point: {left_most}")  # (x, y)
    print(f"Rightmost Point: {right_most}")  # (x, y)

    # Calculate y line for display purposes
    yline = (left_most[1] + right_most[1]) // 2 - int(0.2 * gray.shape[0])

    # Ensure yline is an integer
    yline = int(yline)

    # Draw the y line on the image
    cv2.line(line_image, (0, yline), (line_image.shape[1], yline), (0, 0, 255), 2)  # Red line for y-line

    # Save the Hough Transform result image
    output_hough_path = "hough_output_image.png"
    cv2.imwrite(output_hough_path, line_image)

    print(f"Hough Transform result image saved as {output_hough_path}")

    # Show the result
    # cv2.imshow("Hough Transform Result", line_image)
    # cv2.waitKey(0)  # Wait for a key press to close the window
    # cv2.destroyAllWindows()

    # Return the max width, leftmost, rightmost points, and yline
    return yline


# # Test the function
# image_path = "road_image.png"  # Replace with your image file
# image = cv2.imread(image_path)
#
# # Ensure the image is loaded correctly
# if image is None:
#     print(f"Error: Failed to load image at {image_path}")
# else:
#     # Call the function and get road width, leftmost and rightmost points, and yline
#     road_width, left_point, right_point, yline = get_max_road_width_y(image)
#     print(f"Road width: {road_width} pixels")
#     print(f"Leftmost point: {left_point}")
#     print(f"Rightmost point: {right_point}")
#     print(f"Y-line: {yline}")
