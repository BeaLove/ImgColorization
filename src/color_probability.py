import numpy as np
from pathlib import Path
# from PIL import Image, ImageCms
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import color, io
from tqdm import tqdm


# np.random.seed(0)
# dummy_data = np.random.randint(low=np.min(kernels), high=np.max(kernels), size=(10, 10, 2))
# dummy_data = dummy_data/110

def merge_color_probabilities():
    test_prob = np.load('probabilities test set.npy')
    train_prob = np.load('probabilities training set.npy')
    val_prob = np.load('probabilities val set.npy')
    full_prob = (test_prob + train_prob + val_prob)/3
    print("Sanity check, total probability:", np.sum(full_prob))
    plt.figure()
    plt.plot(full_prob)
    plt.show()
    np.save('full_probabilities.npy', full_prob)

def display_color_probabilities():
    full_prob = np.load('full_probabilities.npy')
    print("Color probability statistics")
    print("Number of bins containing a color:", np.count_nonzero(full_prob > 0))

    plt.figure()
    plt.plot(full_prob)
    plt.show()

def count_ab_colors():
    # batch_size = 150
    POINTS_IN_HULL = np.load('pts_in_hull (1).npy')
    # probabilities_old = np.load('probabilities.npy')
    kernels = POINTS_IN_HULL
    # datalist = Path('../dataset').glob('**/*.TIF')
    # datalist = Path('../dataset/test_tif/').glob('test_tif_1.TIFF')
    # datalist = Path('../dataset/').glob('**/*.JPEG')
    datalist = Path('../dataset/val/').glob('**/*.JPEG')
    # rgb_profile = ImageCms.createProfile('sRGB')
    # lab_profile = ImageCms.createProfile('LAB')
    # rgb_to_lab = ImageCms.buildTransformFromOpenProfiles(rgb_profile,lab_profile,'RGB', 'LAB')
    total_values = np.zeros((21,21))
    for file in tqdm(list(datalist)): #[:batch_size]:
        im = io.imread(file)

        # lab = ImageCms.applyTransform(im, rgb_to_lab)
        if im.shape == (64,64,3):
            lab = color.rgb2lab(im/255)
        else:
            continue
        im = np.array(lab)
        # plt.colorbar()
        # plt.show()
        L, a, b = im[:, :, 0], im[:, :, 1].reshape(-1), im[:, :, 2].reshape(-1)
        values, xedges, yedges = np.histogram2d(a, b, bins=[21, 21], range=[[-110, 100], [-110, 100]])
        total_values += values

    probability = total_values / np.sum(total_values)
    print("Sanity check, total probability:", np.sum(probability))
    plt.figure()
    plt.plot(probability)
    plt.show()
    np.save('probabilities.npy', probability)

    return values

# colors = count_ab_colors()
# merge_color_probabilities()
display_color_probabilities()