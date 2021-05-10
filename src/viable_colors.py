from skimage import io, color
import sklearn.neighbors as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#knn = KNeighborsClassifier(n_neighbors=5)

points = np.load('pts_in_hull (1).npy')

#print(points)

'''find center points of each 10 by 10 bin in the ab color space'''



def getAllRGB():
    all_colors = []
    '''generate all possible rgb colors normalized [0,1]'''
    for r in range(256):
        for g in range(256):
            for b in range(256):
                all_colors.append([r / 255, g / 255, b / 255])
    return np.asarray(all_colors).reshape(4096, 4096, 3)

'''these bin centers form kernels in a 10 by 10 grid (single points that are the bin center rather 
than bins that are ranges of values)'''

def getLabBinKernels():
    bin_centers = np.arange(-110, 115, 10)
    kernels = []
    for a in bin_centers:
        for b in bin_centers:
            kernels.append([a,b])
    return np.asarray(kernels)

def skLearnRGBtoLab(points):
    '''expects a 3d array W*H*3 channels'''
    lab = color.rgb2lab(points)
    l, a, b = lab[:, :, 0], lab[:,:, 1], lab[:,:,2]
    plt.imshow(l)
    plt.show()
    return lab.reshape(4096*4096,3)

'''def getKNearestBins(points, bins, k=5):
    nearest_points = []
    bin_idx = []
    for point in points:
        dist = np.linalg.norm(bins/110 - point, axis=1)
        idx = np.argpartition(dist, kth=5)[:5]
        nearest_bins = bins[idx]
        bin_idx.append(idx)
        nearest_points.append(nearest_bins)
    return nearest_points, bin_idx'''

def countUnique(points):
    bins = np.unique(points)
    return bins

kernels = getLabBinKernels()/110

all_colors = getAllRGB()
print('got all colors')

lab_colors = skLearnRGBtoLab(all_colors)
a,b = lab_colors[:, 1], lab_colors[:, 2]
#frame = pd.DataFrame(lab_colors)
H, xedges, yedges = np.histogram2d(a, b, bins=np.arange(-115, 125, 10))
#ab_vals = frame[[0 == 50/110], 1:]

#lab_colors = np.asarray(ab_vals)

#lab_colors = np.loadtxt('lab_colors.csv', delimiter=',',dtype=float)
#lab_colors.tofile('lab_colors.csv', sep=',')
print('to lab')
#nearest_bins, bin_idx = getKNearestBins(lab_colors, kernels)

neighbors = nn.NearestNeighbors(n_neighbors=5, algorithm='auto').fit(kernels)

dist, indices = neighbors.kneighbors(lab_colors)

num_bins = np.unique(indices)
print('number of unique nearest colors', len(num_bins))
'''
num_bins = np.unique(bin_idx)
bins = nearest_bins[num_bins]
print('number of nearest bins:', len(num_bins))
with open('viable_bins.txt', 'w') as bin_file:

    bin_file.write(str(len(num_bins)))
    bin_file.write('\n')
    bin_file.writelines(str(bins))
print('written to file')'''


'''plt.subplot(1, 2, 1)
plt.imshow(all_colors)'''

'''reshape to a 2d matrix to transform to lab color'''

'''plt.subplot(1, 2, 2)
plt.imshow(lab)
plt.show()
print('converted to lab')'''

'''shape back into a vector for histogram'''


'''fit 5 nearest neighbor classifier to our bin centers'''



'''histogram of a and b color channels in rgb space
H, xedges, yedges = np.histogram2d(a, b, bins=np.arange(-115, 125, 10))

print(np.count_nonzero(H))

X, Y = np.meshgrid(xedges, yedges)

plt.pcolormesh(X, Y, H)
plt.show()'''
