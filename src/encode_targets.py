import numpy as np
import sklearn.neighbors as neighbors
import torch
kernels = np.load('pts_in_hull (1).npy')
print(kernels)
'''load of test code!!!'''
np.random.seed(0)
dummy_data = np.random.randint(low=np.min(kernels), high=np.max(kernels), size=(10, 10, 2))
dummy_data = dummy_data/110
print(dummy_data)
knn = neighbors.NearestNeighbors(n_neighbors=5).fit(kernels/110)
test = np.array([np.arange(10) for i in range(10)])
test2 = np.array([np.arange(10) for i in range(10)])
print(test)
test = test[:,:,np.newaxis]
test2 = test2[:,:,np.newaxis]
test = torch.tensor(np.concatenate((test, test2), axis=2))
result = torch.matmul(test, test)
#result2 = np.dot(test, test)
print(result)
flat = test.reshape(10*10,2)
reshaped = flat.reshape(10,10,2)
def softEncoding(pixels, knn, sigma=5):
    '''args: ground truth pixel values
        out: one hot encoded target vector with 5 nearest points weighted with gaussian kernel'''
    dist, indices = knn.kneighbors(pixels.reshape(pixels.shape[0]*pixels.shape[1], 2))
    weights = np.exp(-(dist**2)/2*sigma**2)
    weights = weights/np.sum(weights, axis=1, keepdims=True)
    '''check weights sum to 1'''
    sum_ = np.sum(weights, axis=1, keepdims=True)
    target_vector = np.zeros((100, 313))
    for i in range(len(weights)):
        target_vector[i, indices[i]] = weights[i]
    test_sum = np.sum(target_vector, axis=1)
    target_vector = target_vector.reshape(10,10,313)
    test_sum2 = np.sum(target_vector, axis=2)
    return target_vector

target = softEncoding(dummy_data, knn)

