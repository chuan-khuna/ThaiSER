from PIL import Image
import numpy as np
import pandas as pd
from .color import Color


def __load_image(image_path: str, max_width: int = 256) -> Image:
    # read file
    img = Image.open(image_path).convert('RGB')
    img_size = np.array(img.size)

    if np.max(img_size) > max_width:

        ratio = (np.max(img_size) / max_width)
        img_size = img_size // ratio
        img_size = np.array(img_size, dtype=int)
        img = img.resize((img_size[0], img_size[1]))

    return img


def __rgb_to_hex(rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    return f"#{r:02x}{g:02x}{b:02x}"


def extract_palette_from_img_array(img_array: np.array,
                                   deltaE_threshold: int | float = 20) -> pd.DataFrame:

    # flatten image
    img_arr = img_array.reshape((np.product(img_array.shape[:2]), 3))

    # count unique colour pixels
    rgb_pixels, counts = np.unique(img_arr, axis=0, return_counts=True)

    # start comparing from most used colour
    sort_id = np.argsort(counts)[::-1]

    rgb_pixels = rgb_pixels[sort_id]
    counts = counts[sort_id]
    num_pixels = len(rgb_pixels)
    groups = [None] * num_pixels
    min_distances = [deltaE_threshold + 1] * num_pixels

    # compare colours
    for i in range(num_pixels):
        hex_c1 = __rgb_to_hex(rgb_pixels[i])
        c1 = Color(hex_c1)

        # if colour group is not assigned
        # assign itself as group
        if groups[i] is None:
            groups[i] = hex_c1
            min_distances[i] = 0

            # find other similar colours
            for j in range(i + 1, num_pixels):
                hex_c2 = __rgb_to_hex(rgb_pixels[j])
                delta_e_dif = c1.delta_e(hex_c2)
                if delta_e_dif < deltaE_threshold and (delta_e_dif < min_distances[j]):
                    groups[j] = hex_c1
                    min_distances[j] = delta_e_dif

    hex_pixels = [__rgb_to_hex(rgb) for rgb in rgb_pixels]
    df = pd.DataFrame({
        'color': hex_pixels,
        'count': counts,
        'nearest_color': groups,
        'min_deltaE': min_distances
    })

    return df


def extract_palette(image_path: str,
                    max_width: int = 128,
                    deltaE_threshold: int | float = 20) -> pd.DataFrame:

    img = __load_image(image_path, max_width)
    img_arr = np.array(img)

    df = extract_palette_from_img_array(img_arr, deltaE_threshold=deltaE_threshold)

    return df


def process_palette_df(df: pd.DataFrame) -> pd.DataFrame:
    group_col = 'nearest_color'
    count_col = 'count'
    rename_to = 'color'
    return df.groupby(group_col).agg({
        count_col: sum
    }).sort_values(by=count_col,
                   ascending=False).reset_index().rename(columns={group_col: rename_to})
