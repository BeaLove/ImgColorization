import numpy as np
import sklearn.neighbors as neighbors
kernels = np.load('pts_in_hull (1).npy')

np.random.seed(0)
dummy_data = np.random.randint(low=np.min(kernels), high=np.max(kernels), size=(10, 2))
dummy_data = dummy_data/110
print(dummy_data)
knn = neighbors.NearestNeighbors(n_neighbors=5).fit(kernels/110)
def softEncoding(pixels, knn, sigma=5):
    '''args: ground truth pixel values
        out: one hot encoded target vector with 5 nearest points weighted with gaussian kernel'''
    dist, indices = knn.kneighbors(pixels)
    weights = np.exp(-(dist**2)/2*sigma**2)
    weights = weights/np.sum(weights, axis=1, keepdims=True)
    '''check weights sum to 1'''
    sum_ = np.sum(weights, axis=1, keepdims=True)
    target_vector = np.zeros((10, 313))
    target_vector[:, indices] = weights
    return target_vector

target = softEncoding(dummy_data, knn)

