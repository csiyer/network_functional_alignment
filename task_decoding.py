"""
Here, we take the subject voxel-to-feature transformation matrices derived in srm.py 
to transform task data into the shared space.

Then, we decode trial conditions (e.g., congruent vs. incongruent) between subjects using leave-one-subject-out cross-validation,
with and without the SRM transformation. In other words, a classifier is trained on SRM-transformed data from all sessions of all
other subjects on a given task. Then, for each trial of the left-out subjects 5 repetitions of that task, it produces a prediction,
which we average to get an accuracy score for that fold. Accuracies are then averaged across left-out subjects.

Assessing the performance benefit of SRM transformation tests how functionally shared or idiosyncratic
neural signatures of these cognitive control tasks are.

Current questions to address:
- Task conditions = ? are we ignoring behavioral success/fail?

Other ideas: 
- Manipulate what data we derive the SRM from. Does GNG-based alignment help decode stop-signal data? Etc.
- Derive SRM from contrast maps instead of rest connectivity?

Author: Chris Iyer
Updated: 10/17/23
"""


### from Brainiak tutorial ###
# Run a leave-one-out cross validation with the subjects
def image_class_prediction(image_data_shared, labels):
    
    subjects = len(image_data_shared)
    train_labels = np.tile(labels, subjects-1)
    test_labels = labels
    accuracy = np.zeros((subjects,))
    for subject in range(subjects):
        
        # Concatenate the subjects' data for training into one matrix
        train_subjects = list(range(subjects))
        train_subjects.remove(subject)
        TRs = image_data_shared[0].shape[1]
        train_data = np.zeros((image_data_shared[0].shape[0], len(train_labels)))
        for train_subject in range(len(train_subjects)):
            start_index = train_subject*TRs
            end_index = start_index+TRs
            train_data[:, start_index:end_index] = image_data_shared[train_subjects[train_subject]]

        # Train a Nu-SVM classifier using scikit learn
        classifier = NuSVC(nu=0.5, kernel='linear')
        classifier = classifier.fit(train_data.T, train_labels)

        # Predict on the test data
        predicted_labels = classifier.predict(image_data_shared[subject].T)
        accuracy[subject] = sum(predicted_labels == test_labels)/len(predicted_labels)
        # Print accuracy
        print("Accuracy for subj %d is: %0.4f" % (subject, accuracy[subject] ))
        
    print("The average accuracy among all subjects is {0:f} +/- {1:f}".format(np.mean(accuracy), np.std(accuracy)))
    return accuracy

# Insert code here
# Load movie data as training
movie_data = np.load(os.path.join(raider_data_dir, 'movie.npy'))

# Load image data and labels as testing
image_data = np.load(os.path.join(raider_data_dir, 'image.npy'))
labels = np.load(os.path.join(raider_data_dir, 'label.npy'))

# convert to list
train_data = []
test_data = []
for sub in range(num_subs):
    train_data.append(movie_data[:,:,sub])  
    test_data.append(image_data[:,:,sub])  
del movie_data, image_data

# Zscore training and testing data

    
# Create the SRM object


# Fit the SRM data
   

# Transform the test data into the shared space using the individual weight matrices


# Zscore the transformed test data


# Run the classification analysis on the test data

