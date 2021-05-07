from skimage import io, color
import sklearn.neighbors as nn
import numpy as np
import matplotlib.pyplot as plt

all_colors = []
#knn = KNeighborsClassifier(n_neighbors=5)

'''find center points of each 10 by 10 bin in the ab color space'''
bin_centers = np.arange(-110, 115, 10)
print(bin_centers)


kernels = []

'''these bin centers form kernels in a 10 by 10 grid (single points that are the bin center rather 
than bins that are ranges of values)'''
for a in bin_centers:
    for b in bin_centers:
        kernels.append([a,b])

print(len(kernels))
kernels = np.asarray(kernels)/110
'''find all possible colors in rgb space'''
for r in range(256):
    for g in range(256):
        for b in range(256):
            all_colors.append([r/255, g/255, b/255])

print('done getting all colors')
all_colors = np.asarray(all_colors).reshape(4096, 4096, 3)
'''plt.subplot(1, 2, 1)
plt.imshow(all_colors)'''

'''reshape to a 2d matrix to transform to lab color'''
lab = color.rgb2lab(all_colors)
'''plt.subplot(1, 2, 2)
plt.imshow(lab)
plt.show()
print('converted to lab')'''

'''shape back into a vector for histogram'''
lab = lab.reshape(4096*4096, 3)

a = lab[:,1]
b = lab[:,2]

ab_vals = lab[:,1:]

'''fit 5 nearest neighbor classifier to our bin centers'''
neighbors = nn.NearestNeighbors(n_neighbors=5, algorithm='auto').fit(kernels)

dist, indices = neighbors.kneighbors(ab_vals)

print(indices.shape)
print(indices)
bins, count = np.unique(indices, return_counts=True)
'''this returns 407 candidate bins...'''
print(len(count))


'''histogram of a and b color channels in rgb space'''
H, xedges, yedges = np.histogram2d(a, b, bins=np.arange(-115, 125, 10))

print(np.count_nonzero(H))

X, Y = np.meshgrid(xedges, yedges)

plt.pcolormesh(X, Y, H)
plt.show()
