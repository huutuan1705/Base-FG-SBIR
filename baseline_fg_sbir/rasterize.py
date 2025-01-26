import scipy.ndimage
import numpy as np

from bresenham import bresenham
from PIL import Image

def get_stroke_num(vector_image):
    return len(np.split(vector_image[:, :2], np.where(vector_image[:, 2])[0] + 1, axis=0)[:-1])