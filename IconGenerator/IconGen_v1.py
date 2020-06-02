from skimage.morphology import convex_hull_image
from skimage import io
from skimage.util import invert
from PIL import Image
import numpy as np
from numpy import asarray


def convex_hull(icon_1_path, icon_2_path, icon_3_path, position_vector):
    icon_1_coords = position_vector[0]  # (x, y) coordinates
    icon_2_coords = position_vector[1]
    icon_3_coords = position_vector[2]

    # Backgrounds and convex hull for icon 1
    background_1 = Image.open("blank_icon.png")  # for convex hull images
    background_1_icon = Image.open("blank_icon.png")  # for real icons

    icon_1 = invert(io.imread('test_icons/console.png', as_gray=True))  # original icon image
    icon_1_chull = asarray(invert(convex_hull_image(icon_1)))  # convex hull icon image
    icon_1_chull_array = asarray(Image.fromarray(icon_1_chull).resize((140, 140)))  # convex hull image as array

    _icon_1 = Image.open('test_icons/console.png').resize((140, 140), Image.ANTIALIAS)
    background_1.paste(Image.fromarray(icon_1_chull_array), (icon_1_coords[0]-70, icon_1_coords[1]-70))
    background_1_icon.paste(_icon_1, (icon_1_coords[0] - 70, icon_1_coords[1] - 70), _icon_1)

    background_1_array = asarray(background_1)  # background w/ pasted chull image as array
    background_1_icon_array = asarray(background_1_icon)

    # Backgrounds and convex hull for icon 2
    background_2 = Image.open("blank_icon.png")  # for convex hull images
    background_2_icon = Image.open("blank_icon.png")  # for real icons

    icon_2 = invert(io.imread('test_icons/food.png', as_gray=True))  # original icon image
    icon_2_chull = asarray(invert(convex_hull_image(icon_2)))  # convex hull icon image
    icon_2_chull_array = asarray(Image.fromarray(icon_2_chull).resize((120, 120)))  # convex hull image as array

    _icon_2 = Image.open('test_icons/food.png').resize((120, 120), Image.ANTIALIAS)
    background_2.paste(Image.fromarray(icon_2_chull_array), (icon_2_coords[0]-60, icon_2_coords[1]-60))
    background_2_icon.paste(_icon_2, (icon_2_coords[0] - 60, icon_2_coords[1] - 60), _icon_2)

    background_2_array = asarray(background_2)  # background w/ pasted chull image as array
    background_2_icon_array = asarray(background_2_icon)

    # Backgrounds and convex hull for icon 3
    background_3 = Image.open("blank_icon.png")  # for convex hull images
    background_3_icon = Image.open("blank_icon.png")  # for real icons

    icon_3 = invert(io.imread('test_icons/heart.png', as_gray=True))  # original icon image
    icon_3_chull = asarray(invert(convex_hull_image(icon_3)))  # convex hull icon image
    icon_3_chull_array = asarray(Image.fromarray(icon_3_chull).resize((120, 120)))  # convex hull image as array

    _icon_3 = Image.open('test_icons/heart.png').resize((120, 120), Image.ANTIALIAS)
    background_3.paste(Image.fromarray(icon_3_chull_array), (icon_3_coords[0]-60, icon_3_coords[1]-60))
    background_3_icon.paste(_icon_3, (icon_3_coords[0]-60, icon_3_coords[1]-60), _icon_3)
    background_3_array = asarray(background_3)  # background w/ pasted chull image as array
    background_3_icon_array = asarray(background_3_icon)

    # Create combined image
    overlap_1_chull = np.where(background_1_array == 0, background_1_array, background_2_array)
    overlap_1_icon = np.where(background_1_array == 0, background_1_icon_array, background_2_icon_array)

    overlap_2_icon = np.where(overlap_1_chull == 0, overlap_1_icon, background_3_icon_array)

    Image.fromarray(overlap_2_icon).show()


convex_hull('test_icons/console.png', 'test_icons/food.png', 'test_icons/heart.png',
            [[100, 100],
             [50, 75],
             [150, 125]])
