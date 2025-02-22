import torchvision.transforms as transforms

def get_transform(type):
    transform_list = [
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    if type == 'train':
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.4),  
            transforms.RandomRotation(15),  
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  
            transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),
        ])
        
    return transforms.Compose(transform_list)