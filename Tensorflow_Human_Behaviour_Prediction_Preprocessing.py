import numpy as np
from sklearn import preprocessing

#Loading the Audiobook data
raw_audiobook_data = np.loadtxt('Audiobooks_data.csv',delimiter=',')

# The inputs are in the columns of the csv except for the first one [:,0] which is just customer IDs 
# last one [:,-1] (which is our targets)
unscaled_inputs_all = raw_audiobook_data[:,1:-1]

# targets_all is taking the target data from the csv which is located in the last column
targets_all = raw_audiobook_data[:,-1]



#BALANCE THE DATASET
# Count how many targets are 1 
# A 1 in the last column represents if the customer converted. A 0 means they did not convert
num_one_targets = int(np.sum(targets_all))

#print(num_one_targets) 

# This is set as a counter for how many targets are 0
zero_targets_counter = 0

# We will have to remove some input/target pairs to create a balanced dataset.
indices_to_remove = []

# Counting the number of targets that are 0  
# Once there are as many 0s as 1s, we will mark the entries where the target is 0 in indices_to_remove
#this will help allow us to have the same number of 0's and 1' 
for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)



#Inputs after we deleted indices that we marked "to remove" in the loop
unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
#Targets after we deleted indices that we marked "to remove" in the loop
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)


#STANDARDIZE THE INPUT
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)



#SHUFFLE THE DATA
# The data is organized in date of purchase
# We will be batching so we need to randomize the data as much as possible
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)
shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]



#SPLITTING DATASET TO TRAIN, VALIDATION AND TEST
# Number of samples
samples_count = shuffled_inputs.shape[0]

# We want 80-10-10 distribution of training, validation, and test
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count

#Splitting data into 
#Training:
train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

#Validation:
validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

# Test:
test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

# We balanced our dataset to be 50-50 (for targets 0 and 1), 
# The training, validation, and test were taken from a shuffled dataset so we will check if they are balanced to be 50-50 as well  
print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)

# Save the three datasets in *.npz.
np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)