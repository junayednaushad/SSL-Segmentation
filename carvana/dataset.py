import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, ids, transform=None, train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.ids = ids
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = np.array(Image.open(os.path.join(self.image_dir, self.ids[idx])))
        if self.train:
            mask = np.array(Image.open(os.path.join(self.mask_dir, self.ids[idx].replace('.jpg', '_mask.gif'))))

        if self.transform:
            if self.train:
                aug = self.transform(image=image, mask=mask)
                image = aug['image']
                mask = aug['mask']
            else:
                aug = self.transform(image=image)
                image = aug['image']
        
        if self.train:
            return image, mask
        else:
            return image

class RotDataset(Dataset):
    def __init__(self, image_dir, ids, transform=None, size=256):
        self.image_dir = image_dir
        self.ids = ids
        self.transform = transform
        self.size = size
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        amount_to_rotate = np.random.randint(0,4)
        image = Image.open(os.path.join(self.image_dir, self.ids[idx]))
        image = image.resize((self.size, self.size))
        image = np.array(image.rotate(90*amount_to_rotate))

        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, amount_to_rotate
