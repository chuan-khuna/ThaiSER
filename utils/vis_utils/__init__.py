# -30k theme

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import patheffects
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

main_colors = {
    "blue": "#202F66",
    "orange": "#FF7048",
    "purple": "#8B87EA",
    "pink": "#D869AB",
    "cyan": "#54C9B9",
    "yellow": "#F3D36E",
}

theme_danger_safe = {
    "danger": "#FF4F72",
    "safe": "#58EDB9",
}

text_colors = {
    "text_black": "#32363A",
    "text_lighter1": "#6F7273",
    "text_lighter2": "#8B8E8F",
    "grey1": "#aeb7bc",
    "grey2": "#c3cace",
    "grey3": "#d9dde0",
    "bg_cream": "#FAF7F4"
}

font_mono = "Incosolata"
font_serif = "Canela Text"
font_sans = "Gill Sans"

# importing theme file from a relative path
plt.style.use(os.path.join(os.path.dirname(__file__), '..', "vis_utils/theme.mplstyle"))

arrows = {'left': '←', 'right': '→', 'up': '↑', 'down': '↓'}

theme_palette = list(main_colors.values())
sns.set_palette(theme_palette)


def theme_path_effects(linewidth=2.5, foreground=text_colors['bg_cream'], alpha=1.0, **kwargs):
    return [patheffects.withStroke(linewidth=linewidth, foreground=foreground, alpha=alpha, **kwargs)]


def mpl_import_fonts(font_paths):
    for font in matplotlib.font_manager.findSystemFonts(font_paths):
        matplotlib.font_manager.fontManager.addfont(font)

def set_font(font_name):
    # set font
    matplotlib.rcParams['font.family'] = font_name
