from skimage.color import *
import numpy as np
import pandas as pd
from typing import TypeVar

ColorType = TypeVar('Color')


class Color:

    DECIMAL_PLACE = 3

    def __init__(self, color_hex: str):
        """_summary_

        Args:
            color_hex (str): color in hex format `#rrggbb`
        """
        self.hex = color_hex

    @property
    def hex(self):
        return self._hex

    @hex.setter
    def hex(self, color_hex: str):
        self._hex = color_hex.upper()
        # update other formats
        self.__to_rgb()
        self.__to_srgb()
        self.__to_luminance()
        self.__to_hsv()
        self.__to_lab()

    def __repr__(self):
        return f"Color hex: {self.hex} luminance {self.luminance}"

    def __to_rgb(self):
        self.rgb = np.array([
            int(f"0x{self.hex[1:3]}", base=16),
            int(f"0x{self.hex[3:5]}", base=16),
            int(f"0x{self.hex[5:]}", base=16)
        ],
                            dtype=int)

    def __to_srgb(self):
        self.sRGB = np.round(self.rgb / 255.0, Color.DECIMAL_PLACE)

    def __to_luminance(self):
        self.luminance = np.round(rgb2gray(self.sRGB), Color.DECIMAL_PLACE)

    def __to_hsv(self):
        # HSB: hue(0-360 degree), saturation(0-100 %), brightness(0-100 %)
        # HSV: hue, saturation, value
        self.hsv = np.round(rgb2hsv(self.sRGB), Color.DECIMAL_PLACE)
        self.hsb = self.hsv

        self.hsv_decimal = np.array(np.round(self.hsv * np.array([360, 100, 100])), dtype=int)
        self.hsb_decimal = self.hsv_decimal

    def __to_lab(self):
        self.lab = rgb2lab(self.sRGB)

    def __to_Color_obj(self, color: str | ColorType):
        """a private function to check and convert input object into Color

        Args:
            color (str | ColorType): string or Color

        Returns:
            _type_: Color obj.
        """
        if isinstance(color, str):
            return Color(color)
        elif isinstance(color, Color):
            return color

    def __calculate_distance(self, color):
        color = self.__to_Color_obj(color)
        return np.round(deltaE_cie76(self.lab, color.lab), Color.DECIMAL_PLACE)

    def __calculate_delta_e(self, color):
        color = self.__to_Color_obj(color)
        return np.round(deltaE_ciede2000(self.lab, color.lab), Color.DECIMAL_PLACE)

    def __calculate_wcag_contrast(self, color):
        color = self.__to_Color_obj(color)

        l1, l2 = self.luminance, color.luminance

        # L2 should be less than L1 in this formular
        if l2 > l1:
            l1, l2 = l2, l1

        return np.round((l1 + 0.05) / (l2 + 0.05), Color.DECIMAL_PLACE)

    def distance(self, color: str | ColorType) -> float:
        """Calculate euclidean distance between this color and the another one

        Args:
            color (str | ColorType): color in string format `#rrggbb` of Color obj.

        Returns:
            _type_: _description_
        """

        return self.__calculate_distance(color)

    def delta_e(self, color: str | ColorType) -> float:
        """calculate delta E (CIEDE 2000) distance between this color and the another one

        delta E in https://en.wikipedia.org/wiki/Color_difference

        Args:
            color (str | ColorType): color in string format `#rrggbb` of Color obj.

        Returns:
            float: _description_
        """
        return self.__calculate_delta_e(color)

    def wcag_contrast(self, color: str | ColorType):
        """calculate WCAG contrast between this color and the another one

        https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html

        Args:
            color (str | ColorType): color in string format `#rrggbb` of Color obj.

        Returns:
            float: _description_
        """
        return self.__calculate_wcag_contrast(color)