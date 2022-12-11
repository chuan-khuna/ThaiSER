import numpy as np
import wordcloud
import pandas as pd
import matplotlib.pyplot as plt


def generate_wordcloud(tokenised_texts):
    words, counts = np.unique(np.concatenate(list(tokenised_texts)), return_counts=True)
    freq_dict = {}
    for k, v in zip(words, counts):
        freq_dict[k] = v
    wc = wordcloud.WordCloud(
        font_path="./fonts/Noto_Sans_Thai/NotoSansThai-VariableFont_wdth,wght.ttf",
        width=2000,
        height=1600,
        max_words=200,
        prefer_horizontal=1.0,
        background_color='white',
        colormap='plasma')
    wc.generate_from_frequencies(freq_dict)

    # plt.imshow(wc)
    # plt.axis(False)
    # plt.show()

    return wc
