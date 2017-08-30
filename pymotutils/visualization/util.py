# vim: expandtab:ts=4:sw=4
import colorsys

import numpy as np
import cv2


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255 * r), int(255 * g), int(255 * b)


def apply_heat_map_uchar(values, mini=None, maxi=None):
    """Color values by their intensity.

    Applies an HSV color map.

    Parameters
    ----------
    values: ndarray
        The N dimensional array of intensities (ndim=1).
    mini : Optional[float]
        The intensity value of minimum saturation (lower bound of color map).
    maxi : Optional[float]
        The intensity value of maximum saturation (upper bound of color map).

    Returns
    -------
    ndarray
        The Nx3 shaped array of color codes in range [0, 255]. The dtype is
        np.int.

    """
    if len(values) == 0:
        return np.zeros((0, ), dtype=np.uint8)
    mini, maxi = mini or np.min(values), maxi or np.max(values)
    valrange = maxi - mini
    if valrange < np.finfo(valrange).eps:
        valrange = np.inf
    normalized = (255. * (values - mini) / valrange).astype(np.uint8)
    colors = cv2.applyColorMap(normalized, cv2.COLORMAP_HSV)
    return colors.astype(np.int).reshape(-1, 3)


def apply_heat_map_float(values, mini=None, maxi=None, dtype=np.float):
    """Color values by their intensity.

    Applies an HSV color map.

    Parameters
    ----------
    values: ndarray
        The N dimensional array of intensities (ndim=1).
    mini : Optional[float]
        The intensity value of minimum saturation (lower bound of color map).
    maxi : Optional[float]
        The intensity value of maximum saturation (upper bound of color map).
    dtype: Optional[nd.dtype]
        Target numeric type for output array.

    Returns
    -------
    ndarray
        The Nx3 shaped array of color codes in range [0, 1].

    """
    if len(values) == 0:
        return np.zeros((0, ), dtype=np.float)
    heat_map_uchar = apply_heat_map_uchar(values, mini, maxi)
    return heat_map_uchar.astype(dtype) / 255.
