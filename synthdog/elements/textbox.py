"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import numpy as np
from synthtiger import layers
from bidi.algorithm import get_display
from arabic_reshaper import reshape

class TextBox:
    def __init__(self, config):
        self.fill = config.get("fill", [1, 1])

    def generate(self, size, text, font):
        width, height = size

        char_layers, chars = [], []
        fill = np.random.uniform(self.fill[0], self.fill[1])
        width = np.clip(width * fill, height, width)
        font = {**font, "size": int(height)}
        # left, top = 0, 0

        # for char in text:
        #     if char in "\r\n":
        #         continue

        #     char_layer = layers.TextLayer(char, **font)
        #     char_scale = height / char_layer.height
        #     char_layer.bbox = [left, top, *(char_layer.size * char_scale)]
        #     if char_layer.right > width:
        #         break

        #     char_layers.append(char_layer)
        #     chars.append(char)
        #     left = char_layer.right
        right, top = width, 0  # Adjust starting position for RTL text
        #reshaped_text = reshape(text)

        for char in text:
            if char in "\r\n":
              continue

            char_layer = layers.TextLayer(char, **font)
            char_scale = height / char_layer.height
            char_layer.bbox = [right - char_layer.width * char_scale, top, char_layer.width * char_scale, height]
            if char_layer.left < 0:  # Adjust boundary condition for RTL text
                break

            char_layers.insert(0, char_layer)  # Insert at the beginning for RTL text
            chars.insert(0, char)  # Insert at the beginning for RTL text
            right = char_layer.left  # Move to the left for the next character

        text = "".join(chars).strip()
        text = get_display(text)
        if len(char_layers) == 0 or len(text) == 0:
            return None, None

        text_layer = layers.Group(char_layers).merge()
        print(text_layer)
        print(text)

        return text_layer, text
