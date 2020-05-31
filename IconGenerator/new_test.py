import numpy as np
import matplotlib.pyplot as plt

from skimage import measure
from skimage import io


# Construct some test data
x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))

# Find contours at a constant value of 0.8
vg = io.imread('love2.png', as_gray=True)
contours = measure.find_contours(vg, 0.1)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(vg, cmap=plt.cm.gray)

ax.plot(contours[0][:, 1], contours[0][:, 0], linewidth=2)

"""for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)"""

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()