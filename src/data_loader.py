import torch
from pathlib import Path
from PIL import Image
import numpy as np


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, paths, stringToReplace, replacementString):

        self.paths = paths
        # targetPaths = [Path(str(path).replace(stringToReplace, replacementString)) for path in self.paths]
        # self.targetPaths = targetPaths
        # print(self.targetPaths[0])
        # print("original path: ", self.paths[0])



        
        

    def __getitem__(self, i):
        colorImg = Image.open(self.paths[i])
        # bwImg = Image.open(self.targetPaths[i])
        # colorImg = np.array(colorImg)
        # bwImg = np.array(bwImg)
        return colorImg
        

    def __len__(self):
        return len(self.paths)



path = Path("./train/")

path = list(path.glob("**/*.JPEG"))

dataset = ConcatDataset(path, "train/", "train_BW/")
color, gs = dataset.__getitem__(546)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=25, num_workers=0, shuffle=True)
x,y = next(iter(train_loader))

print(type(x))
