import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io

# Load the SAR image
sar_img = io.imread('cropped_image2.tif', as_gray=True)

# Reshape the image to a 2D array of pixels
pixels = sar_img.reshape(-1, 1)

# Fit KMeans model to the pixel data
kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
kmeans.fit(pixels)

# Assign each pixel to one of the clusters
segmented_img = kmeans.labels_.reshape(sar_img.shape)

# Show the segmented image
plt.imshow(segmented_img, cmap='gray')
plt.axis('off')
plt.show()
