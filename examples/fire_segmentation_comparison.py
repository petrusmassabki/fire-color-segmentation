import cv2
import numpy as np
import matplotlib.pyplot as plt
import src.fire_segmentation_methods as fsm


rgb_fire_cp = np.load("../data/arrays/phillips_fire_cp_rgb.npy")
hsv_fire_hist = np.load("../data/arrays/bp_fire_histogram_hsv_16_bins.npy")
ycrcb_fire_hist = np.load("../data/arrays/bp_fire_histogram_ycrcb_16_bins.npy")

b_fire_hist = np.load("../data/arrays/rudz_fire_blue_histogram.npy")
g_fire_hist = np.load("../data/arrays/rudz_fire_green_histogram.npy")
r_fire_hist = np.load("../data/arrays/rudz_fire_red_histogram.npy")

image = cv2.imread('../data/fire_images/fire_image_01.png')

methods_params = {"chen": [175, 115],
                  "horng": [30, 55, 170],
                  "celik": [40],
                  "phillips": [rgb_fire_cp],
                  "backprojection_hsv": [hsv_fire_hist, 16, 0.4],
                  "backprojection_ycbcr": [ycrcb_fire_hist, 16, 0.4],
                  "rossi": [4],
                  "rudz": [b_fire_hist, g_fire_hist, r_fire_hist]}

fig = plt.figure(figsize=(10, 10))
fig.add_subplot(3, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image', fontsize=14)
plt.axis('off')

for i, method in enumerate(methods_params.keys()):
    fire_color_segmentation = getattr(fsm, method)
    fire_mask = fire_color_segmentation(image, *methods_params[method])

    fig.add_subplot(3, 3, i + 2)
    plt.imshow(fire_mask, cmap="hot")
    plt.title((" ".join(method.split(sep="_")).title()), fontsize=14)
    plt.axis('off')

plt.tight_layout(pad=0.8)
plt.savefig("fire_segmentation_comparison.png")
