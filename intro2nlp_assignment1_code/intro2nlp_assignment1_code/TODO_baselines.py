# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold
import random

from model.data_loader import DataLoader
import collections
from wordfreq import word_frequency as wf
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.
def safe_div(x,y):
    if y == 0:
        return 0
    return x / y

def majority_baseline(train_sentences, train_labels, testinput, testlabels):
    all_labels = ''.join(train_labels)
    all_labels = all_labels.replace(" ","")
    majority_class = collections.Counter(all_labels).most_common(1)[0]
    majority_class = majority_class[0]
    predictions = []
    for instance in testinput:
        tokens = instance.split(" ")
        instance_predictions = [majority_class for t in tokens]
        predictions.append(instance_predictions)
    #testlabels is a list of stings, while predictions is a list of lists
    # so we convert testlabels into a list of lists
    testlabels = [list(word.replace(" ","").replace("\n","")) for word in testlabels]
    correct = 0
    counter = 0
    for i in range(len(predictions)):
        for j in range(len(testlabels[i])):
            counter = counter + 1
            if testlabels[i][j] == predictions[i][j]:
                correct = correct + 1

        return correct/counter, predictions

def random_baseline(train_sentences, train_labels, testinput, testlabels):
    n_times = 100
    all_labels = ''.join(train_labels)
    all_labels = all_labels.replace(" ","").replace("\n","")
    random_classes = set(all_labels)
    correct = 0
    counter = 0
    # testlabels is a list of stings, while predictions is a list of lists
    # so we convert testlabels into a list of lists
    testlabels = [list(word.replace(" ", "").replace("\n", "")) for word in testlabels]
    for n in range(n_times):
        predictions = []
        for instance in testinput:
            tokens = instance.split(" ")
            instance_predictions = [random.sample(random_classes,1)[0] for t in tokens]
            predictions.append(instance_predictions)
        for i in range(len(predictions)):
            for j in range(len(testlabels[i])):
                counter = counter + 1
                if testlabels[i][j] == predictions[i][j]:
                    correct = correct + 1

            return correct/counter, predictions

def length_baseline(train_sentences, train_labels, testinput, testlabels, threshold):
    all_labels = ''.join(train_labels)
    all_labels = all_labels.replace(" ","").replace("\n","")
    classes = set(all_labels)
    classes = list(classes)
    correct = 0
    counter = 0
    # testlabels is a list of stings, while predictions is a list of lists
    # so we convert testlabels into a list of lists
    testlabels = [list(word.replace(" ", "").replace("\n", "")) for word in testlabels]
    predictions = []
    for instance in testinput:
        tokens = instance.split(" ")
        instance_predictions = [[classes[0], classes[1]][len(t)>threshold] for t in tokens]
        predictions.append(instance_predictions)
    for i in range(len(predictions)):
        for j in range(len(testlabels[i])):
            counter = counter + 1
            if testlabels[i][j] == predictions[i][j]:
                correct = correct + 1

        return correct/counter, predictions

def freq_baseline(train_sentences, train_labels, testinput, testlabels, threshold):
    all_labels = ''.join(train_labels)
    all_labels = all_labels.replace(" ","").replace("\n","")
    classes = set(all_labels)
    classes = list(classes)
    correct = 0
    counter = 0
    # testlabels is a list of stings, while predictions is a list of lists
    # so we convert testlabels into a list of lists
    testlabels = [list(word.replace(" ", "").replace("\n", "")) for word in testlabels]
    predictions = []
    for instance in testinput:
        tokens = instance.split(" ")
        instance_predictions = [[classes[0], classes[1]][wf(t,"en")>threshold] for t in tokens]
        predictions.append(instance_predictions)
    for i in range(len(predictions)):
        for j in range(len(testlabels[i])):
            counter = counter + 1
            if testlabels[i][j]==predictions[i][j]:
                correct = correct + 1

    return safe_div(correct,counter),predictions

def flat_list(list):
    flat_list = []
    for sublist in list:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def flat_list_with_n(list):
    flat_list = []
    for sublist in list:
        for item in sublist:
            if not item=="\n" and not item==" ":
                flat_list.append(item)
    return flat_list

if __name__ == '__main__':
    train_path = "data/preprocessed/train/"
    dev_path = "data/preprocessed/val/"
    test_path = "data/preprocessed/test/"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.

    with open(train_path + "sentences.txt") as sent_file:
        train_sentences = sent_file.readlines()

    with open(train_path + "labels.txt") as label_file:
        train_labels = label_file.readlines()

    with open(dev_path + "sentences.txt") as dev_file:
        dev_sentences = dev_file.readlines()

    with open(dev_path + "labels.txt") as dev_label_file:
        dev_labels = dev_label_file.readlines()

    with open(test_path + "sentences.txt") as testfile:
        testinput = testfile.readlines()

    with open(test_path + "labels.txt") as test_label_file:
        testlabels = test_label_file.readlines()

    print("test data")
    majority_accuracy_test, majority_predictions = majority_baseline(train_sentences, train_labels, testinput, testlabels)
    random_accuracy_test, random_predictions = random_baseline(train_sentences, train_labels, testinput, testlabels)
    length_accuracy_test, length_predictions = length_baseline(train_sentences, train_labels, testinput, testlabels,2)
    freq_accuracy_test, freq_predictions = freq_baseline(train_sentences, train_labels, testinput,testlabels, 0.00008)
    print("accuracy")
    print(majority_accuracy_test)
    print(random_accuracy_test)
    print(length_accuracy_test)
    print(freq_accuracy_test)

    print("dev data")
    majority_accuracy_dev,  majority_predictions = majority_baseline(train_sentences, train_labels, dev_sentences, dev_labels)
    random_accuracy_dev,  random_predictions = random_baseline = random_baseline(train_sentences, train_labels, dev_sentences, dev_labels)
    length_accuracy_dev, length_predictions = length_baseline(train_sentences, train_labels, dev_sentences, dev_labels,2)
    freq_accuracy_dev,  freq_predictions = freq_baseline(train_sentences, train_labels, dev_sentences, dev_labels,0.00008)
    print("accuracy")
    print(majority_accuracy_dev)
    print(random_accuracy_dev)
    print(length_accuracy_dev)
    print(freq_accuracy_dev)

    print("recall")
    print(recall_score(flat_list_with_n(dev_labels),flat_list(random_predictions),average=None))
    print(recall_score(flat_list_with_n(dev_labels), flat_list(majority_predictions), average=None))
    print(recall_score(flat_list_with_n(dev_labels), flat_list(length_predictions), average=None))
    print(recall_score(flat_list_with_n(dev_labels), flat_list(freq_predictions), average=None))
    print("percision")
    print(precision_score(flat_list_with_n(dev_labels),flat_list(random_predictions),average=None))
    print(precision_score(flat_list_with_n(dev_labels), flat_list(majority_predictions), average=None))
    print(precision_score(flat_list_with_n(dev_labels), flat_list(length_predictions), average=None))
    print(precision_score(flat_list_with_n(dev_labels), flat_list(freq_predictions), average=None))
    print("f1")
    print(f1_score(flat_list_with_n(dev_labels),flat_list(random_predictions),average=None))
    print(f1_score(flat_list_with_n(dev_labels), flat_list(majority_predictions), average=None))
    print(f1_score(flat_list_with_n(dev_labels), flat_list(length_predictions), average=None))
    print(f1_score(flat_list_with_n(dev_labels), flat_list(freq_predictions), average=None))
    print("f1 weighted")
    print(f1_score(flat_list_with_n(dev_labels),flat_list(random_predictions),average="weighted"))
    print(f1_score(flat_list_with_n(dev_labels), flat_list(majority_predictions), average="weighted"))
    print(f1_score(flat_list_with_n(dev_labels), flat_list(length_predictions), average="weighted"))
    print(f1_score(flat_list_with_n(dev_labels), flat_list(freq_predictions), average="weighted"))

    print("report")
    print("random")
    print(classification_report(flat_list_with_n(dev_labels),flat_list(random_predictions)))
    print("majority")
    print(classification_report(flat_list_with_n(dev_labels), flat_list(majority_predictions)))
    print("length")
    print(classification_report(flat_list_with_n(dev_labels), flat_list(length_predictions)))
    print("freq")
    print(classification_report(flat_list_with_n(dev_labels), flat_list(freq_predictions)))