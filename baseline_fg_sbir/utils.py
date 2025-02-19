import torchvision.transforms as transforms

def get_transform(type):
    transform_list = [
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4821, 0.4465], [0.2471, 0.2435, 0.2616])
    ]
    
    if type == 'train':
        transform_list.extend([
            transforms.RandomHorizontalFlip(0.1),
            transforms.RandomRotation(10),
            transforms.RandomCrop(32, padding=2)
        ])
        
    return transforms.Compose(transform_list)
    