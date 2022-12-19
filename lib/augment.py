import torchvision.transforms as transforms
import numpy as np  

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

class SpatialConsistentColorAug:
    """Take two random crops of one image"""

    def __init__(self, crop_min):
        self.base_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_min, 1.)),
            transforms.RandomHorizontalFlip()])
        self.color_aug = transforms.Compose([transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize])
        self.to_tensor = transforms.Compose([transforms.ToTensor(),
                normalize])

    def __call__(self, x):
        im = self.base_transform(x)
        im1 = self.color_aug(im)
        im2 = self.color_aug(im)
        return [im1, im2]

def create_augmentation(args):
    
    if args.aug_centercrop:
        augmentation = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]

    if args.aug_spatialconsistent_color:
        return SpatialConsistentColorAug(args.crop_min)

    if args.aug_spatial:
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ] 

    return transforms.Compose(augmentation)
