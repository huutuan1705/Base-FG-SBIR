import scipy.ndimage
import numpy as np

from bresenham import bresenham
from PIL import Image

# đếm và trả về số nét vẽ (stroke) trong một hình ảnh vector dựa trên cột cờ xác định điểm bắt đầu của từng nét
def get_stroke_num(vector_image):
    return len(np.split(vector_image[:, :2], np.where(vector_image[:, 2])[0] + 1, axis=0)[:-1])

def draw_image_from_list(vector_image, stroke_idx, side=256):
    vector_image = np.split(vector_image[:, :2], np.where(vector_image[:, 2])[0] + 1, axis=0)[:-1]
    vector_image = [vector_image[x] for x in stroke_idx]
    
    raster_image = np.zeros((int(side), int(side)), dtype=np.float32)
    
    for stroke in vector_image:
        initX, initY = int(stroke[0, 0]), int(stroke[0, 1])
        
        for i_pos in range(1, len(stroke)):
            cord_list = list(bresenham(initX, initY, int(stroke[i_pos, 0]), int(stroke[i_pos, 1])))
            for cord in cord_list:
                if (cord[0] > 0 and cord[1] > 0) and (cord[0] <= side and cord[1] <= side):
                    raster_image[cord[1], cord[0]] = 255.0
                else:
                    print('error')
                    
            initX, initY = int(stroke[i_pos, 0]), int(stroke[i_pos, 1])
            
    raster_image = scipy.ndimage.binary_dilation(raster_image)*255.0
    return Image.fromarray(raster_image).convert('RGB')

def draw_image(vector_image, side=256):
    raster_image = np.zeros((int(side), int(side)), dtype=np.float32)
    initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])