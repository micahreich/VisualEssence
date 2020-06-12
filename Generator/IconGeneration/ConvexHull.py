from skimage.morphology import convex_hull_image
from skimage import io
from skimage.util import invert
from PIL import Image
import numpy as np
from numpy import asarray


def convex_hull(path_to_data, icons, position_vector, n_icons=3):

    icon_arrays = []
    for i in range(n_icons):
        # /nethome/mreich8/VisualEssence/
        background = Image.open(path_to_data[:path_to_data.index("VisualEssence")] + "VisualEssence/Generator/IconGeneration/blank_icon.png")  # for convex hull images
        background_icon = Image.open(path_to_data[:path_to_data.index("VisualEssence")] + "VisualEssence/Generator/IconGeneration/blank_icon.png")  # for real icons

        icon = invert((255-asarray(Image.open(path_to_data + "/" + icons[i])))[:, :, 3])  # original icon image
        icon_chull = asarray(invert(convex_hull_image(icon)))  # convex hull icon image
        icon_chull_array = asarray(Image.fromarray(icon_chull).resize((120, 120)))  # convex hull image as array

        _icon = Image.open(path_to_data + "/" + icons[i]).resize((120, 120), Image.ANTIALIAS)
        background.paste(Image.fromarray(icon_chull_array), (position_vector[i][0]-60, position_vector[i][1]-60))
        background_icon.paste(_icon, (position_vector[i][0]-60, position_vector[i][1]-60), _icon)

        background_array = asarray(background)  # background w/ pasted chull image as array
        background_icon_array = asarray(background_icon)

        icon_arrays.append([background_array, background_icon_array])

    # Create combined image
    overlap_1_chull = np.where(icon_arrays[0][0] == 0, icon_arrays[0][0], icon_arrays[1][0])
    overlap_1_icon = np.where(icon_arrays[0][0] == 0, icon_arrays[0][1], icon_arrays[1][1])

    overlap_2_icon = np.where(overlap_1_chull == 0, overlap_1_icon, icon_arrays[2][1])

    image_final = Image.fromarray(overlap_2_icon)

    return image_final

