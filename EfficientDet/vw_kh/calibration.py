import cv2
import numpy as np
import matplotlib.pyplot as plt

CHECKERBOARD = "./checkerboard.png"

CHECK_X = 13
CHECK_Y = 9

CHECK_DISTANCE = 30  # cm
CHECK_GAP = 2  # cm

image = cv2.imread(CHECKERBOARD)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_h, image_w = image.shape
retval, corners = cv2.findChessboardCorners(image, (CHECK_X, CHECK_Y))
if not retval:
    raise ValueError("Checkerboard corners are not found...")

# corners = list of (x, y)
sorted_corners = list(sorted([cc[0] for cc in corners], key=lambda c: c[1]))
actual_points = []
for y in range(CHECK_Y):
    for x in range(CHECK_X):
        actual_points.append([(CHECK_X // 2 - x) * CHECK_GAP, (CHECK_Y // 2 - y) * CHECK_GAP])

# print(sorted_corners)
# print(actual_points)
assert len(sorted_corners) == len(actual_points)

sorted_corners = np.array(sorted_corners)  # (N, 2)
actual_points = np.array(actual_points, dtype=np.float32)  # (N, 2)
print(sorted_corners.shape, actual_points.shape)

valid_x_min = int(np.min(sorted_corners[:, 0]) + 16)
valid_x_max = int(np.max(sorted_corners[:, 0]) - 15)
valid_y_min = int(np.min(sorted_corners[:, 1]) + 16)
valid_y_max = int(np.max(sorted_corners[:, 1]) - 15)

actual_coords = np.zeros((image_h, image_w, 2), dtype=np.float32)  # (h, w, 2)

for y in range(valid_y_min, valid_y_max):
    for x in range(valid_x_min, valid_x_max):
        coord = np.array([x, y], dtype=np.float32)
        coord_diff = np.sum((sorted_corners - coord) * (sorted_corners - coord), axis=-1)  # (N)
        coord_sort = np.argsort(coord_diff)

        c1, p1 = sorted_corners[coord_sort[0]], actual_points[coord_sort[0]]
        c2, p2 = sorted_corners[coord_sort[1]], actual_points[coord_sort[1]]
        c3, p3 = sorted_corners[coord_sort[2]], actual_points[coord_sort[2]]
        c4, p4 = sorted_corners[coord_sort[3]], actual_points[coord_sort[3]]

        a1 = np.prod(np.abs(coord - c1)) + 1e-5
        a2 = np.prod(np.abs(coord - c2)) + 1e-5
        a3 = np.prod(np.abs(coord - c3)) + 1e-5
        a4 = np.prod(np.abs(coord - c4)) + 1e-5
        a_sum = (1 / a1) + (1 / a2) + (1 / a3) + (1 / a4)
        w1 = 1 / (a1 * a_sum)
        w2 = 1 / (a2 * a_sum)
        w3 = 1 / (a3 * a_sum)
        w4 = 1 / (a4 * a_sum)

        p = p1 * w1 + p2 * w2 + p3 * w3 + p4 * w4
        actual_coords[y, x] = p

plt.figure()
plt.imshow(actual_coords[:, :, 0])
plt.show()
plt.figure()
plt.imshow(actual_coords[:, :, 1])
plt.show()

angles = np.ones((image_h, image_w), dtype=np.float32) * (-1.0)
for y in range(valid_y_min, valid_y_max):
    for x in range(valid_x_min, valid_x_max):
        d = np.sqrt(np.sum(actual_coords[y, x] * actual_coords[y, x]))
        deg = np.rad2deg(np.arctan2(d, CHECK_DISTANCE))
        angles[y, x] = deg

print(angles.max(), angles.min())
plt.figure()
plt.imshow(angles[:, :])
plt.show()
np.save("degree.npy", angles)
