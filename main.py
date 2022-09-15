import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import argparse
import os

# Create argparse argument for command line usability
parser = argparse.ArgumentParser(description='Short program that recognizes hand gestures from Sign Language MNIST - '
                                             'The American Sign Language letter database of hand gestures '
                                             'represent a multi-class problem with 24 classes of letters'
                                             '(excluding J and Z which require motion')
parser.add_argument('train_path', help='path to train CSV file with train dataset')
parser.add_argument('test_path', help='path to test CSV file with test dataset')
parser.add_argument('-p', '--plot_examples', help='plot input examples', action='store_true')
parser.add_argument('-H', '--history', help='plot train and test accuracy and loss', action='store_true')
parser.add_argument('-v', '--verbosity', help='increase output verbosity', action='store_true')
args = parser.parse_args()


# Import train and test datasets from CSV
train_path = args.train_path
test_path = args.train_path

#train_path = 'sign_mnist_train.csv'
#test_path = 'sign_mnist_test.csv'
def import_from_csv(path):

    """
    Imports *.csv file, no. of examples x [label, pixel1, pixel2... pixel754]
    Args:
        path - path to a data *.csv file
    Returns:
         m - no. of examples
         n - no. of features (pixels)
         X - input data, m x n_w pixel x n_h pixel x 1
         Y - label vector of size m x 1
    """
    data = pd.read_csv(train_path).to_numpy()
    X = data[:, 1:]
    X = X.astype(float)
    Y = data[:, 0]
    m, n = X.shape  # m - no. of examples, n - no. of features
    X = np.reshape(X, (m, np.sqrt(n).astype(int), np.sqrt(n).astype(int), 1))

    return m, n, X, Y

# Plot few examples
def plot_examples(X: np.ndarray, Y: np.ndarray):
    """
    Plots 'random' examples from ndarray X (examples, pixel, pixel, 1) and
    corresponding labels from ndarray Y (examples x label)

    Args:
        X - np.ndarray (m x pix x pix x 1)
        Y - np.ndarray Y (m x 1)
    Returns:
        None
    """
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(X[i+j, :, :], cmap='inferno')
            axs[i, j].set_title(str(Y[i+j]))
            axs[i, j].axis('off')
    plt.show()

if __name__=='__main__':

    # Import and normalize pictures
    m_train, n_train, X_train, Y_train = import_from_csv(train_path)
    m_test, n_test, X_test, Y_test = import_from_csv(train_path)

    #Plot examples of train dataset
    if args.plot_examples == True:
        plot_examples(X_train, Y_train)

    X_train /= 255.0
    X_test /= 255.0

    uniq_labels = np.unique(Y_train)
    #print(uniq_labels)

    depth = len(uniq_labels) + 1  # Because if the missing label 9

    Y_train = tf.keras.utils.to_categorical(
        Y_train, num_classes=depth, dtype='float32')

    Y_test = tf.keras.utils.to_categorical(
        Y_test, num_classes=depth, dtype='float32')


    # Defining class for callbacks
    class accuracy_clbk(tf.keras.callbacks.Callback):

        """
        Halts the training after reaching 99 percent accuracy
        Args:
        epoch (integer) - index of epoch (required but unused in the function definition below)
        logs (dict) - metric results from the training epoch
        """

        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') >= 0.99:
                print('\nReached 99% accuracy, cancelling training!')
                self.model.stop_training = True


    # Instantiate class
    callback = accuracy_clbk()

    # Define the classification model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(depth, activation=tf.nn.softmax)
    ])

    # Setup training parameters
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # Train the model
    print(f'\nMODEL TRAINING:')
    model.fit(X_train, Y_train, epochs=20, callbacks=[callback])

    test_loss = model.evaluate(X_test, Y_test)

    if args.verbosity == 1:
        print(f'The model works with {test_loss[1]*100:.2f} % of accuracy on the test set.')
    else:
        print(f'Test set score: {test_loss}')
        # Test set score: [0.04015837237238884, 0.9892187118530273]
