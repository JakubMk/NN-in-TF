import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

# Import train and test datasets from CSV
train_path = 'sign_mnist_train.csv'
test_path = 'sign_mnist_test.csv'


def import_from_csv(path):
    data = pd.read_csv(train_path).to_numpy()
    X = data[:, 1:]
    X = X.astype(float)
    Y = data[:, 0]
    m, n = X.shape  # m - no. of examples, n - no. of features
    X = np.reshape(X, (m, 28, 28, 1))

    return m, n, X, Y


# Plot few examples
def plot_examples():
    fig, axs = plt.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(X[i + j, :, :, 1], cmap='inferno')
            axs[i, j].set_title(str(Y[i + j]))
            axs[i, j].axis('off')
    plt.show()


# Import and normalize pictures
m_train, n_train, X_train, Y_train = import_from_csv(train_path)
m_test, n_test, X_test, Y_test = import_from_csv(train_path)
# plot_examples() # Uncomment if you want to plot 8 examples
X_train /= 255.0
X_test /= 255.0

# To make sure classes are randomly distributed
# plt.plot(Y_train)
# plt.show()

uniq_labels = set()

for i in range(len(Y_train)):
    uniq_labels.add(Y_train[i])

print(uniq_labels)

depth = len(uniq_labels) + 1  # Because if the missing label 9

Y_train = tf.keras.utils.to_categorical(
    Y_train, num_classes=depth, dtype='float32')

Y_test = tf.keras.utils.to_categorical(
    Y_test, num_classes=depth, dtype='float32')


# Defining class for callbacks
class myCallback(tf.keras.callbacks.Callback):

    # Halts the training after reaching 60 percent accuracy

    # Args:
    #  epoch (integer) - index of epoch (required but unused in the function definition below)
    #  logs (dict) - metric results from the training epoch

    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') >= 0.99):
            print('\nReached 60% accuracy, cancelling training!')
            self.model.stop_training = True


# Instantiate class
callback = myCallback()

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
model.fit(X_train, Y_train, epochs=20, callbacks = [callback])

test_loss = model.evaluate(X_test, Y_test)
print(f'Test set score: {test_loss}')
# Test set score: [0.04015837237238884, 0.9892187118530273]
