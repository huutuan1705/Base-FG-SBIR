import cv2
import os

# Đường dẫn folder chứa ảnh gốc và folder lưu ảnh kết quả
input_folder = 'D:/Research/Sketch_based_image_retrieval/dataset/ChairV2/sketch'    # Thay bằng đường dẫn folder chứa ảnh gốc
output_folder = 'D:/Research/Sketch_based_image_retrieval/dataset/ChairV2/new_sketch'  # Thay bằng đường dẫn folder lưu ảnh kết quả

# Tạo folder đích nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Lặp qua từng file trong folder ảnh gốc
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Đọc ảnh gốc
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Đảo ngược màu: nền đen, nét trắng
        inverted_image = cv2.bitwise_not(image)

        # Lưu ảnh kết quả với định dạng PNG
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, inverted_image)

        print(f"Đã xử lý: {filename}")

print("Hoàn thành chuyển đổi tất cả ảnh!")
