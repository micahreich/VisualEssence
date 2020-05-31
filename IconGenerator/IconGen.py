import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image
from skimage import data, img_as_float
from skimage import io
from skimage.util import invert
from numpy import asarray

# The original image is inverted as the object must be white.
image = invert(io.imread('love2.png', as_gray=True))
print(asarray(image).shape)

chull = convex_hull_image(image)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

ax[0].set_title('Original picture')
ax[0].imshow(invert(image), cmap=plt.cm.gray)
ax[0].set_axis_off()

ax[1].set_title('Transformed picture')
ax[1].imshow(invert(chull), cmap=plt.cm.gray)
ax[1].set_axis_off()

plt.tight_layout()
plt.show()