from steganalysis import fuse_image_features
from joblib import load
from joblib import dump
from sklearn import svm
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from imgaug import augmenters as iaa
import imageio
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time

def main():
    def import_images_from_folder(folder_path):
        images = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(folder_path, filename)
                images.append(img_path)
        return images

    # Specify the folder paths
    cover_folder_path = "cover"
    stego_folder_path = "stego"

    # Import images from the folders
    cover_images = import_images_from_folder(cover_folder_path)
    stego_images = import_images_from_folder(stego_folder_path)

    # Label the images
    cover_labels = [0] * len(cover_images)  # 0 for "unmodified"
    stego_labels = [1] * len(stego_images)  # 1 for "modified"

    # Combine the images and labels
    images = cover_images + stego_images
    labels = cover_labels + stego_labels

    # Extract the features from the images
    features_array = fuse_image_features(images, labels)
    

    if not os.path.exists('svm_model.joblib'):
        # Initialize the classifier
        clf = SVC()

        # Define the parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 1, 10],
            'kernel': ['rbf', 'linear', 'poly']
        }

        # Initialize the grid search
        grid_search = GridSearchCV(clf, param_grid, cv=5, verbose=2, n_jobs=-1)

        # Perform the grid search
        grid_search.fit(features_array, labels)

        # Get the best estimator
        clf = grid_search.best_estimator_

        # Print the best parameters
        print(f"Best parameters: {grid_search.best_params_}")
    else: 
        # Load the trained model
        clf = load('svm_model.joblib')
        print("Model loaded")

    # Train the classifier
    clf.fit(features_array, labels)
    
    #Save the trained model
    dump(clf, 'svm_model.joblib')

    #Specify the folder path for testing
    folder_path = "test"

    # Import images from the folder
    images = import_images_from_folder(folder_path)

    # Create dummy labels
    labels = [0] * len(images)

    # Call the fuse_image_features function
    features_array = fuse_image_features(images, labels)

    # Load the trained model
    clf = load('svm_model.joblib')

    # Classify images using the loaded classifier
    predictions = clf.predict(features_array)

    # Print the predictions
    for image, prediction in zip(images, predictions):
        print(f"Image: {image}, Prediction: {prediction}")

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution Time:", execution_time, "seconds")



    # # Define the augmentation sequence
    # seq = iaa.Sequential([
    #     iaa.Fliplr(0.5),  # horizontal flips
    #     # ... (rest of the augmenters)
    # ], random_order=True)

    # # Apply the augmentation sequence to each image in the training set
    # images_augmented = []
    # for i, image in enumerate(images):
    #     augmented_image = seq(image=imageio.imread(image))
    #     augmented_image_path = f"augmented_{i}.png"
    #     imageio.imsave(augmented_image_path, augmented_image)
    #     images_augmented.append(augmented_image_path)

    # # Extract the features from the augmented images
    # features_array_augmented = fuse_image_features(images_augmented, labels)

    # # Extract the features from the augmented images
    # features_array_augmented = fuse_image_features(images_augmented, labels)

#     if not os.path.exists('gradient_boosting_model.joblib'):
#         # Initialize the classifier
#         clf = RandomForestClassifier(n_estimators=100)
#     else: 
#         # Load the trained model
#         clf = load('random_forest_model.joblib')
#         print("Model loaded")

#     # Train the classifier
#     clf.fit(features_array, labels)

#     # Save the trained model
#     dump(clf, 'random_forest_model.joblib1')

#     # # Perform 5-fold cross validation
#     # scores = cross_val_score(clf, features_array_augmented, labels, cv=5)

#     # # Print the mean score and the 95% confidence interval
#     # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#     # Specify the folder path for testing
#     folder_path = "test"

#     # Import images from the folder
#     images = import_images_from_folder(folder_path)

#     # Create dummy labels
#     labels = [0] * len(images)

#     # Call the fuse_image_features function
#     features_array = fuse_image_features(images, labels)

#     # Load the trained model
#     clf = load('random_forest_model.joblib1')

#     # Classify images using the loaded classifier
#     predictions = clf.predict(features_array)

#     # Print the predictions
#     for image, prediction in zip(images, predictions):
#         print(f"Image: {image}, Prediction: {prediction}")

# if __name__ == '__main__':
#     main()


#     from sklearn.ensemble import GradientBoostingClassifier

#     if not os.path.exists('gradient_boosting_model.joblib'):
#         # Initialize the classifier
#         clf = GradientBoostingClassifier(n_estimators=100)
#     else: 
#         # Load the trained model
#         clf = load('gradient_boosting_model.joblib')
#         print("Model loaded")

#     # Train the classifier
#     clf.fit(features_array, labels)

#     # Save the trained model
#     dump(clf, 'gradient_boosting_model.joblib')

#     # Specify the folder path for testing
#     folder_path = "test"

#     # Import images from the folder
#     images = import_images_from_folder(folder_path)

#     # Create dummy labels
#     labels = [0] * len(images)

#     # Call the fuse_image_features function
#     features_array = fuse_image_features(images, labels)

#     # Load the trained model
#     clf = load('gradient_boosting_model.joblib')

#     # Classify images using the loaded classifier
#     predictions = clf.predict(features_array)

#     # Print the predictions
#     for image, prediction in zip(images, predictions):
#         print(f"Image: {image}, Prediction: {prediction}")

# if __name__ == '__main__':
#     main()