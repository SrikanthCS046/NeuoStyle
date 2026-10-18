import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
class ImageFolderDataset(Dataset):
    def __init__(self,root,transform=None):
        super(ImageFolderDataset,self).__init__()
        self.root=root
        self.transform=transform
        self.files=list(os.listdir(self.root))
        self.files=[p for p in self.files if p.endswith(("jpg","png","jpeg"))]

    def __len__(self):
        return len(self.files) #it will return the length of the dataset

    def __getitem__(self, index):
        image_path=os.path.join(self.root,self.files[index])
        #read the image using image library
        image=Image.open(image_path).convert("RGB")
        if self.transform:
            image=self.transform(image)
        return image


def get_transform(size, crop, final_size):
    transform_list = []

    if size > 0:
        transform_list.append(
            transforms.Resize((size, size))
        )

    if crop:
        transform_list.append(
            transforms.RandomCrop((final_size, final_size))
        )
    else:
        transform_list.append(
            transforms.Resize((final_size, final_size))
        )

    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)


def adaptive_instance_normalization(c_feat,s_feat):
    size=c_feat.size()
    style_mean,style_std=calc_mean_std(s_feat)
    content_mean,content_std=calc_mean_std(c_feat)
    normalized_content_feat=(c_feat-content_mean.expand(size))/content_std.expand(size)
    return normalized_content_feat*style_std.expand(size)+style_mean.expand(size)


def calc_mean_std(feat,eps=1e-5):
    size=feat.size()
    assert(len(size)==4)
    batch_size,channels=size[:2]
    feat_mean=feat.view(batch_size,channels,-1).mean(dim=2).view(batch_size,channels,1,1)
    feat_var=feat.view(batch_size,channels,-1).var(dim=2)+eps
    feat_std=feat_var.sqrt().view(batch_size,channels,1,1)
    return feat_mean,feat_std


