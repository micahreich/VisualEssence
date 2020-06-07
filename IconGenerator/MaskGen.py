from PIL import Image
import numpy as np


def make_row_outline(path_to_image):
    img_arr = np.asarray(Image.open(path_to_image))
    img_arr = ((255 - np.asarray(img_arr))[:, :, 3])
    black = np.zeros(img_arr.shape)
    white = np.ones(img_arr.shape) * 255

    img_arr = np.where(img_arr < 255, black, white)
    outline = np.ones(img_arr.shape) * 255

    for r in range(len(img_arr)):
        total_black_px_count = list(img_arr[r]).count(0)
        black_px_count = 0
        for c in range(len(img_arr[0])):
            if img_arr[r][c] < 255 and black_px_count == 0:
                black_px_count += 1
                outline[r][c] = 0
            elif img_arr[r][c] < 255 and black_px_count == total_black_px_count-1:
                outline[r][c] = 0
            elif img_arr[r][c] < 255:
                black_px_count += 1

    mask = np.ones(img_arr.shape) * 255

    for r in range(len(outline)):
        black_px_count = 0
        for c in range(len(outline)):
            if outline[r][c] < 255:
                black_px_count += 1
            if outline[r][c] == 255 and black_px_count == 1:
                mask[r][c] = 0

    return mask


def make_col_outline(path_to_image):
    img_arr = np.asarray(Image.open(path_to_image))
    img_arr = ((255 - np.asarray(img_arr))[:, :, 3])
    black = np.zeros(img_arr.shape)
    white = np.ones(img_arr.shape) * 255

    img_arr = np.where(img_arr < 255, black, white)
    img_arr = np.rot90(img_arr, k=1)

    outline = np.ones(img_arr.shape) * 255

    for r in range(len(img_arr)):
        total_black_px_count = list(img_arr[r]).count(0)
        black_px_count = 0
        for c in range(len(img_arr[0])):
            if img_arr[r][c] < 255 and black_px_count == 0:
                black_px_count += 1
                outline[r][c] = 0
            elif img_arr[r][c] < 255 and black_px_count == total_black_px_count-1:
                outline[r][c] = 0
            elif img_arr[r][c] < 255:
                black_px_count += 1

    mask = np.ones(img_arr.shape) * 255

    for r in range(len(outline)):
        black_px_count = 0
        for c in range(len(outline)):
            if outline[r][c] < 255:
                black_px_count += 1
            if outline[r][c] == 255 and black_px_count == 1:
                mask[r][c] = 0

    mask = np.rot90(mask, k=3)
    return mask


def ExpandMask(input, iters):
    """
    Expands the True area in an array 'input'.

    Expansion occurs in the horizontal and vertical directions by one
    cell, and is repeated 'iters' times.
    """
    yLen, xLen = input.shape
    output = input.copy()
    for iter in range(iters):
        for y in range(yLen):
            for x in range(xLen):
                if (y > 0 and not input[y - 1, x]) or \
                        (y < yLen - 1 and not input[y + 1, x]) or \
                        (x > 0 and not input[y, x - 1]) or \
                        (x < xLen - 1 and not input[y, x + 1]): output[y, x] = 0
        input = output.copy()
    return output

def generate_mask(path_to_image, min_max):
    col_mask = make_col_outline(path_to_image)
    row_mask = make_row_outline(path_to_image)

    col_mask_area = np.count_nonzero(col_mask == 0)
    row_mask_area = np.count_nonzero(row_mask == 0)

    if min_max == 0:
        if col_mask_area < row_mask_area:
            return col_mask
        else:
            return row_mask
    elif min_max == 1:
        if col_mask_area < row_mask_area:
            return row_mask
        else:
            return col_mask
