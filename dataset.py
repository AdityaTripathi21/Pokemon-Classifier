import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

df = pd.read_csv("data/pokemon.csv")
print(df.columns)
img_dir = "data/images"
files = os.listdir(img_dir)
print(len(files))
print(files[:20])


# Must implement len and get item
class PokemonDataset(Dataset):
    def __init__(self, img_df, images_dir, transform=None):
        super().__init__()
        self.df = img_df
        self.images_dir = images_dir
        self.transform = transform

        types = pd.concat([self.df["Type1"], self.df["Type2"]], ignore_index=True)    # stack series on top of each other
        types = types.dropna().astype(str).str.lower().str.strip()  # drop missing values, convert to string, lowercase and strip
        self.all_types = sorted(types.unique())  # returns unique values only
        self.types_index = {t: i for i, t in enumerate(self.all_types)}   # dict for type -> index


    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # multi-hot encoding 

        row = self.df.iloc[index]   # use iloc for integer location of index
                                    # row is a panda series with one value per column

        y = torch.zeros(len(self.all_types), dtype=torch.float32)   # tensor of 0s

        t1 = row["Type1"]
        t2 = row["Type2"]

        if pd.notna(t1):
            y[self.types_index[str(t1).lower().strip()]] = 1.0  # specific index becomes 1.0
        if pd.notna(t2):
            y[self.types_index[str(t2).lower().strip()]] = 1.0  # specific index becomes 1.0

        name = row["Name"]  # get image name
        img_name = str(name).lower().strip() + ".png"   # convert name to name.png
        img_path = os.path.join(self.images_dir, img_name)  # get full path

        image = Image.open(img_path).convert("RGB") # open the file path and then convert to 3 channels for RGB
                                                    # the mode is RGBA before conversion

        if self.transform is not None:      # apply transform if it exists
            image = self.transform(image)   # need transform for converting to tensor, reordering dimensions, 
                                            # normalizing pixels and resizing images


        return image, y



