import os
import numpy as np
import cv2
from model import Model
import numpy as np
from Layers import *  
from Loss import * 
from Optimizer import * 
from accuracy import * 



def down_load_data ():
    from zipfile import ZipFile
    import os
    import urllib
    import urllib.request

    URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
    FILE = 'fashion_mnist_images.zip'
    FOLDER = 'fashion_mnist_images'
    if not os.path.isfile(FILE):
        print(f'Downloading {URL} and saving as {FILE}...')
        urllib.request.urlretrieve(URL, FILE)
    print('Unzipping images...')
    with ZipFile(FILE) as zip_images:
        zip_images.extractall(FOLDER)
    print('Done!')


# MNIST dataset (train + test)
def create_data_mnist(path):
    # down_load_data()
    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    # And return all the data
    return X, y, X_test, y_test


# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):
    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))
    # Create lists for samples and labels
    X = []
    y = []
    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            # And append it and a label to the lists
            X.append(image)
            y.append(label)
            # Convert the data to proper numpy arrays and return
    
    return np.array(X), np.array(y).astype('uint8')



# Create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]


# Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

# Instantiate the model
model = Model()


# Add layers
model.add(Dense(X.shape[1], 128))
model.add(ReLU())
model.add(Dense(128, 128))
model.add(ReLU())
model.add(Dense(128, 10))
model.add(Softmax())

# Set loss, optimizer and accuracy objects
model.set(
    loss=CategoricalCrossentropy(),
    optimizer=Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test),
            epochs=10, batch_size=128, print_every=100)
