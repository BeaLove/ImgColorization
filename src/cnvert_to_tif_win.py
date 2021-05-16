from pathlib import Path
from PIL import Image, ImageOps, ImageCms
from tqdm import tqdm
import os
from skimage import color, io
import matplotlib.pyplot as plt
""" 
Extract the dataset (folders test, train, val) and place them into folder dataset.
Then run this script to convert the .JPEG images to .TIF images.

https://stackoverflow.com/questions/52767317/how-to-convert-rgb-image-pixels-to-lab/53353542#53353542
"""


def convert_to_tif():
    root = os.path.dirname(os.path.abspath(__file__)).replace('\\\\', '/') + '\\'
    os.makedirs(root + '..\\dataset\\train_tif\\', exist_ok=True)
    os.makedirs(root + '..\\dataset\\val_tif\\', exist_ok=True)
    os.makedirs(root + '..\\dataset\\test_tif\\', exist_ok=True)
    paths = [root + '..\\dataset\\test\\', root + '..\\dataset\\val\\', root + '..\\dataset\\train\\']
    skip_count = 0
    for path in paths:
        #path = Path(path)
        #pic_list = list(path.glob('**/*.JPEG'))
        pic_list = []
        for dir, subdir, files in os.walk(path, topdown=False, followlinks=False):
            for f in files:
                pic_list.append(dir + "\\" + f)
            print(pic_list[0])
        #srgb_p = ImageCms.createProfile("sRGB")
        #lab_p = ImageCms.createProfile("LAB")
        #rgb_to_lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")

        for pic in tqdm(pic_list):
            try:
                im = io.imread(pic)
            except ValueError:
                skip_count += 1
                continue
            #with io.imread(pic) as im:
           #print('hello world')
            try:
                tif_img = color.rgb2lab(im/255)
            except ValueError:
                skip_count += 1
                continue

            pic_str = str(pic)

            if os.name == 'nt':  # If windows
                pic_str.replace('\\', '/')

            word = pic_str.replace(root, '').split('\\')[2]
            save_path = pic_str.replace(word, word + '_tif')
            save_path = save_path.replace('.JPEG', '.TIF')
            save_path_dir = ''.join([w + '\\' for w in save_path.split('\\')[:-1]])
            os.makedirs(save_path_dir, exist_ok=True)
            io.imsave(save_path, tif_img)
    print("num images skipped", skip_count)
            #plt.imshow(tif_img[:,:,2])
            #plt.show()
            #lab_scaled = (tif_img + [0, 128, 128]) / [100, 255, 255]
            #io.imshow(lab_scaled)
            #io.show()
            #tif_img.save(save_path, 'TIFF')


if __name__ == '__main__':
    convert_to_tif()