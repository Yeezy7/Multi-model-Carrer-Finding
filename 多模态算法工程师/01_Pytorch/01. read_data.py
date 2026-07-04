import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir) 
        self.img_path = os.listdir(self.path) # 列表下的所有图像文件名


    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    
    def __len__(self):
        return len(self.img_path)
        
    

if __name__ == "__main__":
    dataset = MyData(root_dir="/Users/ikun/Pictures", label_dir="snipaste")
    print(len(dataset))
    img, label = dataset[0]
    print(img)
    print(label)
    # img.show()
    
    ## dataloader
    print("===============dataloader===============")
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
    
    for img, label in dataloader:
        print(img)
        print(label)
        break
    
    
    
    