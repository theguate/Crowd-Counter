import base64
import re
from io import BytesIO

import cv2
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from PIL import Image


def transform_images(image):
    transform = standard_transforms.Compose(
        [standard_transforms.ToTensor(),
         standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225]), ])
    img = Image.fromarray(image).convert('RGB')
    img = torch.Tensor(transform(img)).unsqueeze(0)
    return img


def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


def pred_map_overlay(pred_map, pred_cnt, rgb_color, text_color, bg_color):
    density_map = cv2.normalize(pred_map.detach().numpy()[0, 0, :, :], None, 0, 255, cv2.NORM_MINMAX)
    density_map = cv2.cvtColor(density_map, cv2.COLOR_GRAY2RGB)
    text = str(int(pred_cnt))
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
    x, y = 75, 75
    cv2.rectangle(density_map, (x, y - size[0][1] - size[1]), (x + size[0][0], y + size[0][1] - size[1]), bg_color, cv2.FILLED)
    cv2.putText(density_map, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 2)
    return density_map


def hex_to_rgb(hx, hsl=False):
    """Converts a HEX code into RGB or HSL.
    Args:
        hx (str): Takes both short as well as long HEX codes.
        hsl (bool): Converts the given HEX code into HSL value if True.
    Return:
        Tuple of length 3 consisting of either int or float values.
    Raise:
        ValueError: If given value is not a valid HEX code."""
    if re.compile(r'#[a-fA-F0-9]{3}(?:[a-fA-F0-9]{3})?$').match(hx):
        div = 255.0 if hsl else 0
        if len(hx) <= 4:
            return tuple(int(hx[i]*2, 16) / div if div else
                         int(hx[i]*2, 16) for i in (1, 2, 3))
        return tuple(int(hx[i:i+2], 16) / div if div else
                     int(hx[i:i+2], 16) for i in (1, 3, 5))
    raise ValueError(f'"{hx}" is not a valid HEX code.')


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
