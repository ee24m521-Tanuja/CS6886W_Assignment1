from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from augmentations import Cutout
MEAN=(0.4914,0.4822,0.4465); STD=(0.2023,0.1994,0.2010)
def get_loaders(batch_size=64, data_dir='./data', workers=2):
    train_tf=transforms.Compose([transforms.RandomCrop(32,padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(MEAN,STD), Cutout(1,16)])
    test_tf=transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    tr=datasets.CIFAR10(data_dir,train=True,download=True,transform=train_tf)
    te=datasets.CIFAR10(data_dir,train=False,download=True,transform=test_tf)
    common=dict(batch_size=batch_size,num_workers=workers,pin_memory=True,persistent_workers=workers>0)
    return DataLoader(tr,shuffle=True,**common),DataLoader(te,shuffle=False,**common)
