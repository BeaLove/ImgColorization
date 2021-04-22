from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
import os


""" 
Extract the dataset (folders test, train, val) and place them into folder dataset.
Then run this script to convert the .JPEG images to .TIF images.
"""


def turn_BW():
    os.makedirs('./train_BW/', exist_ok = True)
    os.makedirs('./validation_BW/', exist_ok = True)
    os.makedirs('./test_BW/', exist_ok = True)
    paths = ['./test/', './val/', './train/']
    
    for path in paths:
        path = Path(path)
        pic_list = list(path.glob('**/*.JPEG'))

        for pic in tqdm(pic_list):
            with Image.open(pic) as im:
                gs = ImageOps.grayscale(im)
                first_word = str(pic).split('/')[0]
                save_path = str(pic).replace(first_word+'/' , first_word + '_BW/')
                save_path_dir = ''.join([word+'/' for word in save_path.split('/')[:-1]])
                os.makedirs(save_path_dir, exist_ok = True)
                gs.save(save_path, 'JPEG')

turn_BW()

# 'test/images/test_3736.JPEG'
# 'test_BW/images/test_3736.JPEG'
