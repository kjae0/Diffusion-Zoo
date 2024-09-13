from torchvision.transforms import transforms

def get_transform(cfg):
    if cfg['transform'] == 'default':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise NotImplementedError(f"Unknown transform type: {cfg['transform']}")
