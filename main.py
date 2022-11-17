from Dictionary_maker import create_dictionary
from Features_extractor import extract_features
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score
import time
from datetime import timedelta

start = time.time()

TRAIN_DIR = "C:/Users/parth/PycharmProjects/nnfs_L3/train-mails"
TEST_DIR = "C:/Users/parth/PycharmProjects/nnfs_L3/test-mails"

# TRAIN_DIR = "C:/Users/parth/PycharmProjects/nnfs_L3/enron1/train"
# TEST_DIR = "C:/Users/parth/PycharmProjects/nnfs_L3/enron1/test"


dictionary_size = 3000
dictionary = create_dictionary(TRAIN_DIR,dictionary_size)


features_matrix,labels = extract_features(TRAIN_DIR,dictionary,dictionary_size)
test_feature_matrix, test_labels = extract_features(TEST_DIR,dictionary,dictionary_size)

model = GaussianNB()
print("Training model...")

model.fit(features_matrix, labels)
predicted_labels = model.predict(test_feature_matrix)

print("FINISHED classifying. accuracy score : ")
print(accuracy_score(test_labels, predicted_labels))


end = time.time()
diff =  end-start
print("elapsed time: "+str(timedelta(microseconds=diff)))





