import numpy as np
from pathlib import Path
# from PIL import Image, ImageCms
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import color, io
from tqdm import tqdm
import misc.npy_loader.loader as npy

# np.random.seed(0)
# dummy_data = np.random.randint(low=np.min(kernels), high=np.max(kernels), size=(10, 10, 2))
# dummy_data = dummy_data/110

def merge_color_probabilities():
    test_prob = npy.load('probabilities_test_set')
    train_prob = npy.load('probabilities_train_set')
    val_prob = npy.load('probabilities_val_set')
    full_prob = (test_prob + train_prob + val_prob)/3
    print("Sanity check, total probability:", np.sum(full_prob))
    plt.figure()
    plt.plot(full_prob)
    plt.show()
    npy.save('full_probabilities', full_prob)

def display_color_probabilities():
    test = npy.load('filtered_probabilities_gaussian_sigma_5')
    full_prob = npy.load('full_probabilities')
    print("Color probability statistics")
    print("Number of bins containing a color:", np.count_nonzero(full_prob > 0))
    bins = np.arange(-105, 100, 10)
    non_zero_bins = np.nonzero(full_prob)
    valid_kernels = [[bins[a], bins[b]] for a,b in zip(non_zero_bins[0], non_zero_bins[1])]
    npy.save('bins', valid_kernels)
    plt.figure()
    plt.plot(full_prob.reshape(-1))
    plt.show()

def gaussian_filter():
    full_prob = npy.load('full_probabilities')
    select = np.where(full_prob > 0, full_prob)
    from scipy.ndimage.filters import gaussian_filter
    filtered = gaussian_filter(full_prob.reshape(-1), sigma = 5)
    npy.save('filtered_probabilities_gaussian', filtered)
    # filtered = full_prob.reshape(-1)
    plt.figure()
    plt.plot(filtered)
    plt.savefig('Filtered probabilities - gaussian.png')

def uniform_distribution():
    filtered_prob = npy.load('filtered_probabilities_gaussian_sigma_5')
    weight = 1/((filtered_prob * 0.5) + 0.5/len(filtered_prob))
    sum = np.sum(filtered_prob * weight)
    weight = weight / np.sum(weight)
    print("Weight: ", weight)
    print("Expectation:", np.sum(filtered_prob * weight))
    npy.save('weight_distribution_mix_with_uniform_distribution_normalized', weight)

def count_ab_colors():
    # batch_size = 150
    POINTS_IN_HULL = npy.load('authors_pts_in_hull')
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
    # np.save('probabilites.npy', probability)

    return values

def bin_centers():
    probs = npy.load('filtered_probabilities_gaussian_sigma_5')
    bins = np.arange(-105, 100, 10)
    kernels = []
    for a in bins:
        for b in bins:
            kernels.append([a,b])
    kernels = np.asarray(kernels)
    npy.save('bin_centers', kernels)
    useful_probs = np.where(probs > 0)[0]
    good_kernels = kernels[useful_probs]
    return kernels

def get_valid_probs():
    full_probs = npy.load('full_probabilities')
    select = np.where(full_probs>0, full_probs)
    filtered = gaussian_filter(select.reshape(-1), sigma=5)
    weight = 1 / ((filtered * 0.5) + 0.5 / len(filtered))
    sum = np.sum(filtered * weight)
    weight = weight / np.sum(weight)
    npy.save('weights_normalized', weight)
# colors = count_ab_colors()
# merge_color_probabilities()
display_color_probabilities()
get_valid_probs()
#gaussian_filter()
#uniform_distribution()
#bin_centers()