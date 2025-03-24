import os
import cv2
import scipy.ndimage
import numpy as np
import pickle

from bresenham import bresenham
from PIL import Image

# đếm và trả về số nét vẽ (stroke) trong một hình ảnh vector dựa trên cột cờ xác định điểm bắt đầu của từng nét
def get_stroke_num(vector_image):
    return len(np.split(vector_image[:, :2], np.where(vector_image[:, 2])[0] + 1, axis=0)[:-1])

def draw_image_from_list(vector_image, stroke_idx, side=400):
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

def draw_image(vector_image, side=400):
    raster_image = np.zeros((int(side), int(side)), dtype=np.float32)
    initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
    pixel_length = 0

    for i in range(0, len(vector_image)):
        if i > 0:
            if vector_image[i - 1, 2] == 1:
                initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

        cordList = list(bresenham(initX, initY, int(vector_image[i, 0]), int(vector_image[i, 1])))
        pixel_length += len(cordList)

        for cord in cordList:
            if (cord[0] > 0 and cord[1] > 0) and (cord[0] < side and cord[1] < side):
                raster_image[cord[1], cord[0]] = 255.0
        initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

    raster_image = scipy.ndimage.binary_dilation(raster_image) * 255.0
    return raster_image

def preprocess(sketch_points, side=400):
    sketch_points = sketch_points.astype(np.float32)
    sketch_points[:, :2] = sketch_points[:, :2] / np.array([256, 256])
    sketch_points[:, :2] = sketch_points[:, :2] * side
    sketch_points = np.round(sketch_points)
    return sketch_points

def rasterize_sketch(sketch_points):
    sketch_points = preprocess(sketch_points)
    raster_images = draw_image(sketch_points)
    return raster_images

def rasterize_sketch_save(sketch_points, save_path=None, save_name=""):
    sketch_points = preprocess(sketch_points)
    raster_images = draw_image(sketch_points)
    if save_path:
        save_full_path = f"{save_path}/{save_name}"
        # Sử dụng OpenCV
        # cv2.imwrite(save_full_path, raster_images)

        # Hoặc dùng PIL
        image = Image.fromarray(raster_images.astype(np.uint8))
        image.save(save_full_path)

        print(f"Đã lưu ảnh tại: {save_full_path}")
    

if __name__ == "__main__":
    root_dir = "D:\Research\Sketch_based_image_retrieval\dataset"
    dataset_name = "ShoeV2"
    save_path = "D:/Research/Sketch_based_image_retrieval/dataset/ShoeV2/sketchs"
    coordinate_path = os.path.join(root_dir, dataset_name, dataset_name + '_Coordinate')
    with open(coordinate_path, 'rb') as f:
        coordinate = pickle.load(f)
    
    save_folder = "sketch_images"
    train_sketch = [x for x in coordinate if 'train' in x]
    test_sketch = [x for x in coordinate if 'test' in x]
    
    for item in range(len(train_sketch)):
        save_name = train_sketch[item].split('/')[-1] + ".jpg"
        # print(save_name)
        sketch_path = train_sketch[item]
        vector_x = coordinate[sketch_path]
        rasterize_sketch_save(vector_x, save_path, save_name) 