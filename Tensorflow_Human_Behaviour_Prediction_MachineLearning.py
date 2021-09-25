import numpy as np
import tensorflow as tf

#DATA
# let's store each of the three Audiobooks datasets [inputs,targets]
# targets must be integers because of sparse_categorical_crossentropy 
npz = np.load('Audiobooks_data_train.npz')
train_inputs = npz['inputs'].astype(np.float)
train_targets = npz['targets'].astype(np.int)

# we load the validation data in the temporary variable
npz = np.load('Audiobooks_data_validation.npz')
# we can load the inputs and the targets in the same line
validation_inputs = npz['inputs'].astype(np.float)
validation_targets =  npz['targets'].astype(np.int)

# we load the test data in the temporary variable
npz = np.load('Audiobooks_data_test.npz')
test_inputs = npz['inputs'].astype(np.float)
test_targets = npz['targets'].astype(np.int)


#MODEL
# The model will have an input layer of 10 units
# There will be two hidden layers each with 50 units
# There will be an output layer with two output nodes (two outputs, 0 and 1)
input_size = 10
output_size = 2
hidden_layer_size = 50
    
# defining what the model will look like
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    tf.keras.layers.Dense(output_size, activation='softmax') # output layer
])


# Choosing the optimizer and the loss function
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training with a set batch size of 100
# setting a maximum number of training epochs of 100
batch_size = 100
max_epochs = 100

# set an early stopping mechanism
# let's set patience=2, to be a bit tolerant against random validation loss increases
early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

# fit the model
model.fit(train_inputs, 
          train_targets, 
          batch_size=batch_size, 
          epochs=max_epochs, 
          callbacks=[early_stopping], 
          validation_data=(validation_inputs, validation_targets), 
          verbose = 2 
          )  