#%% Importing libraries
import numpy as np
import matplotlib.pyplot as plt

#%% Additional functions

def load_image(path):
    '''
    Function loads image in a .png or .jpeg (.jpg) format and transform it to 
    2D array with RGB values for each pixel.
    
    Parameters
    ----------
    path : string 
        Path to the picture file.

    Returns
    -------
    X_img : 2D ndarray with RGB values for each pixel of the picture.
        Copy of original picture.

    picture_orig : 2D ndarray with RGB values for each pixel of the picture.
        Original (without compression) picture.
    original_shape : tupple (m, n)
        Shape (size in pixels) of the original picture.
    '''
    picture_orig = plt.imread(path)
    picture_orig = picture_orig/255
    original_shape = picture_orig.shape
    X_img = np.reshape(picture_orig, (picture_orig.shape[0]*picture_orig.shape[1], 3))
    
    return X_img, picture_orig, original_shape


def initial_centroids (X, K):
    '''
    Function establishes initial centroids as randomly chosen K pixels 
    from the given picture.

    Parameters
    ----------
    X : 2D ndarray (m, n)
        Array of m pixels and n features. For each pixel there are three values 
        corresponding with RGB channels.
    K : int
        Number of initial centroids to be established.

    Returns
    -------
    init_centroids : 2D ndarray (K, n)
        Array of K centroids represented by RGB coordinates.
    '''
    rng = np.random.default_rng()
    idx = rng.permutation(X.shape[0])
    init_centroids = X[idx[:K],]
    
    return init_centroids


def closest_centroids(X, centroids):
    '''
    Function computes distance (L2 norm) between pixels and centroids and find
    the closest centroid for each pixel.

    Parameters
    ----------
    X : 2D ndarray (m, n)
        Array of m pixels and n features. For each pixel there are three values 
        corresponding with RGB channels.
    centroids : 2D ndarray (K, n)
        Array of K centroids represented by RGB coordinates.

    Returns
    -------
    CC_idx : 1D ndarray
        Array with indexes of the closest centroids for each pixel.
    '''
    m = X.shape[0]
    K = centroids.shape[0]
    CC_idx = np.zeros(X.shape[0], dtype = int)
    for i in range(m):
        odleglosc = []
        for j in range(K):
            diff = X[i]-centroids[j]
            odleglosc.append(np.dot(diff, diff))
        CC_idx[i] = np.argmin(odleglosc)
    
    return CC_idx


def new_centroids(X, CC_idx, K):
    '''
    Function computes new centroid as a mean of pixels assigned to the centroid.

    Parameters
    ----------
    X : 2D ndarray (m, n)
        Array of m pixels and n features. For each pixel there are three values 
        corresponding with RGB channels.
    CC_idx : 1D ndarray
        Array with indexes of the closest centroids for each pixel.
    K : int
        Number of centroids.

    Returns
    -------
    new_C : 2D ndarray (K, n)
        Array of K centroids represented by RGB coordinates. There may be 'nan'
        values in case of centroid without any pixel assigned.

    '''
    new_C = np.empty((0,3), int)
    for l in range(K):
        piksele = X[CC_idx==l]
        ilosc_pikseli = piksele.shape[0]
        suma = np.sum(piksele, axis=0)
        if ilosc_pikseli != 0:
            new_C = np.append(new_C, [suma/ilosc_pikseli], axis=0)
        else:
            new_C = np.append(new_C,[[np.nan, np.nan, np.nan]], axis=0)
            
    return new_C

#%% Main function

def run_K_means (path, iterations, K):
    '''
    Function performs compression of the given picture.

    Parameters
    ----------
    path : string
        Path to the picture file.
    iterations : int
        Number of clustering iterations.
    K : int
        Number of centroids.

    Returns
    -------
    recovered_picture : 2D ndarray with RGB values for each pixel of the picture.
        Picture after compression with K-means algorithm.

    '''
    X, original_picture, original_shape = load_image(path)
    init = initial_centroids(X, K)
    m, n = X.shape
    
    for i in range(iterations):
        CC_idx = closest_centroids(X, init)
        newest_C = new_centroids(X, CC_idx, K)
    
    recovered_picture = newest_C[CC_idx, :] 
    recovered_picture = np.reshape(recovered_picture, original_shape)
    
    return original_picture, recovered_picture

def visualise_images(original_picture, recovered_picture):
    '''
    Function visualise effects of compression.

    Parameters
    ----------
    original_picture : 2D ndarray with RGB values for each pixel of the picture.
        Original (without compression) picture.
    recovered_picture : 2D ndarray with RGB values for each pixel of the picture.
        Picture after compression with K-means algorithm.

    '''
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(original_picture)
    ax1.set_title("Original picture")
    ax2.imshow(recovered_picture)
    ax2.set_title("Compressed picture")
    
    return None

original_picture, recovered_picture = run_K_means("images/Kitty.jpg", 20, 30)
visualise_images(original_picture, recovered_picture)