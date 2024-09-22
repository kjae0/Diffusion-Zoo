from torchvision.transforms import transforms

def get_transform(cfg):
    if cfg['transform'] == 'default':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ])
    elif cfg['transform'] == 'celeba':
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ])
    else:
        raise NotImplementedError(f"Unknown transform type: {cfg['transform']}")
