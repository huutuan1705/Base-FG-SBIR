import torchvision.transforms as transforms
import torch

def get_transform(type):
    transform_list = [
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    if type == 'train':
        transform_list = [
            transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),  # Cắt ngẫu nhiên
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Dịch chuyển ảnh
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Điều chỉnh màu sắc
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # Làm mờ ảnh ngẫu nhiên
        ]
        
    return transforms.Compose(transform_list)