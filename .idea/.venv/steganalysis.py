import os
import numpy as np
from PIL import Image
from scipy import stats
from skimage import feature
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def extract_images_from_folder(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
    images = []

    for file_name in os.listdir(folder_path):
        if any(file_name.lower().endswith(ext) for ext in image_extensions):
            images.append(os.path.join(folder_path, file_name))

    return images

# Get the current working directory
current_dir = os.getcwd()

# Relative path to your image folder
relative_folder_path = 'attachments'

# Join the current directory with the relative folder path
folder_path = os.path.join(current_dir, relative_folder_path)

images = extract_images_from_folder(folder_path)

for image_path in images:
    image = Image.open(image_path)
    image_matrix = np.array(image)
    print(image_matrix)

#RS analysis

def median_edge_detector(image_matrix):
    # Create a copy of the image matrix to store the predicted values
    predicted_matrix = image_matrix.copy()

    # Iterate over the image matrix and apply the MED predictor formula
    for i in range(1, image_matrix.shape[0]):
        for j in range(1, image_matrix.shape[1]):
            predicted_matrix[i, j] = np.median([
                image_matrix[i-1, j],
                image_matrix[i, j-1],
                image_matrix[i-1, j] + image_matrix[i, j-1] - image_matrix[i-1, j-1]
            ])

    return predicted_matrix

def calculate_residuals(image_matrix, predicted_matrix):
    # Calculate the residuals
    return image_matrix - predicted_matrix

def analyze_residuals(residuals):
    # Perform statistical analysis on the residuals
    print("Mean:", np.mean(residuals))
    print("Standard deviation:", np.std(residuals))
    print("Skewness:", stats.skew(residuals.flatten()))
    print("Kurtosis:", stats.kurtosis(residuals.flatten()))

for image_path in images:
    image = Image.open(image_path)
    image_matrix = np.array(image)
    predicted_matrix = median_edge_detector(image_matrix)
    residuals = calculate_residuals(image_matrix, predicted_matrix)
    analyze_residuals(residuals)

#LB analysis

def calculate_lbp(image_matrix, points=8, radius=1):
    # Calculate the Local Binary Pattern representation of the image
    lbp = feature.local_binary_pattern(image_matrix, points, radius, method="uniform")
    return lbp

labels = ['modified', 'unmodified']  


for image_path in images:
    image = Image.open(image_path)
    image_matrix = np.array(image)
    for image_path, label in zip(images, labels):
        image = Image.open(image_path)
        image_matrix = np.array(image)
        # Convert the image to grayscale if it's colored
        if len(image_matrix.shape) > 2:
            image_matrix = rgb2gray(image_matrix)
            image_matrix = img_as_ubyte(image_matrix)  # convert to uint8
    

    lbp = calculate_lbp(image_matrix)
    print('lbp', lbp)

    
def calculate_histogram(lbp):
    # Calculate the histogram of the LBP values
    lbp_hist, _ = np.unique(lbp.ravel(), return_counts=True)
    # Normalize the histogram
    lbp_hist = normalize(lbp_hist.reshape(1, -1), norm='l1')
    return lbp_hist

def extract_features(lbp_hist, residuals):
    # Extract features from the LBP histogram and the residuals
    features = np.concatenate((lbp_hist, residuals), axis=None)
    return features

def fuse_features(features_rs, features_lbp):
    # Reshape the features to 2D
    features_rs = np.reshape(features_rs, (1, -1))
    features_lbp = np.reshape(features_lbp, (1, -1))

    # Fuse the features using PCA
    pca = PCA(n_components=min(len(features_rs), len(features_lbp)))
    fused_features = pca.fit_transform(np.concatenate((features_rs, features_lbp), axis=0))
    return fused_features


for image_path in images:
    image = Image.open(image_path)
    image_matrix = np.array(image)
    predicted_matrix = median_edge_detector(image_matrix)
    residuals = calculate_residuals(image_matrix, predicted_matrix)
    analyze_residuals(residuals)
    for image_path in images:
        image = Image.open(image_path)
        residuals = np.array(image)
        for image_path, label in zip(images, labels):
            image = Image.open(image_path)
            residuals = np.array(image)
            # Convert the image to grayscale if it's colored
            if len(residuals.shape) > 2:
                residuals = rgb2gray(residuals)
                residuals = img_as_ubyte(residuals)  # convert to uint8
        

        lbp = calculate_lbp(residuals)

    # Calculate LBP on the residuals
    lbp = calculate_lbp(residuals)
    print('lbp', lbp)

    # Calculate the histogram of the LBP values
    lbp_hist = calculate_histogram(lbp)

    # Extract features from the LBP histogram and the residuals
    features = extract_features(lbp_hist, residuals)

    # Fuse the features
    fused_features = fuse_features(features, features)

    # Use SVM for classification
    clf = svm.SVC()
    clf.fit(fused_features, labels)
    predicted_labels = clf.predict(fused_features)
    print('Accuracy:', accuracy_score(labels, predicted_labels))
    print('Precision:', precision_score(labels, predicted_labels, average='weighted'))
    print('Recall:', recall_score(labels, predicted_labels, average='weighted'))
    print('F1 Score:', f1_score(labels, predicted_labels, average='weighted'))

# if np.var(features_rs) < 1e-10 or np.var(features_lbp) < 1e-10:
#     print('Warning: The variance of the features is too low for PCA.')
# else:
#     # Fuse the features using PCA
#     pca = PCA(n_components=min(len(features_rs), len(features_lbp)))
#     fused_features = pca.fit_transform(np.concatenate((features_rs, features_lbp), axis=0))

# # Use SVM for classification
# clf = svm.SVC()

# # Check if the model is predicting all labels
# unique_labels = np.unique(labels)
# clf.fit(fused_features, labels)
# predicted_labels = clf.predict(fused_features)
# if not np.all(np.isin(unique_labels, predicted_labels)):
#     print('Warning: The model is not predicting all labels.')
# else:
#     print('Accuracy:', accuracy_score(labels, predicted_labels))
#     print('Precision:', precision_score(labels, predicted_labels, average='weighted'))
#     print('Recall:', recall_score(labels, predicted_labels, average='weighted'))
#     print('F1 Score:', f1_score(labels, predicted_labels, average='weighted'))