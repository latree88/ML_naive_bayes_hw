import numpy as np
import random
import os
import math

fold_num = 10
c_parameter = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

def LoadSpamData(filename = "spambase.data"):
    # """
    # Each line in the datafile is a csv with features values, followed by a single label (0 or 1),
    # per sample; one sample per line
    # """

    # "The file function reads the filename from the current directory, unless you provide an absolute path
    # e.g. /path/to/file/file.py or C:\\path\\to\\file.py"

    unprocessed_data_file = file(filename,'r')

    "Obtain all lines in the file as a list of strings."

    unprocessed_data = unprocessed_data_file.readlines()

    labels = []
    features = []

    for line in unprocessed_data:
        feature_vector = []

        "Convert the String into a list of strings, being the elements of the string separated by commas"
        split_line = line.split(',')

        "Iterate across elements in the split_line except for the final element "
        for element in split_line[:-1]:
            feature_vector.append(float(element))

        "Add the new vector of feature values for the sample to the features list"
        features.append(feature_vector)

        "Obtain the label for the sample and add it to the labels list"
        labels.append(int(split_line[-1]))

    "Return List of features with its list of corresponding labels"
    return features, labels




def ConvertDataToArrays(the_features):
    """
    conversion to a numpy array is easy if you're starting with a List of lists.
    The returned array has dimensions (M,N), where M is the number of lists and N is the number of

    """

    return np.asarray(the_features)


def get_mean_and_std(the_training):
    the_means = np.mean(the_training, axis = 0)
    the_std = np.std(the_training, axis = 0)

    return the_means, the_std



def attachLabel(the_features, the_labels):
    for i in range(len(the_features)):
        the_features[i].append(the_labels[i])
    return the_features


def split(the_dataSet):
    # half_size = len(dataSet)/2
    # the_training = dataSet[:half_size]
    # the_testing = dataSet[half_size:]

    the_training = []
    the_testing = []

    pos_index = 0
    for i in range(len(the_dataSet)):
        if the_dataSet[i][-1] == 1:
            pos_index = i
    half_size_pos = pos_index/2
    half_size_neg = (len(the_dataSet) - pos_index) / 2

    the_training_pos = dataSet[:half_size_pos]
    the_testing_pos = dataSet[half_size_pos:pos_index + 1]
    the_training_neg = dataSet[(pos_index + 1):(half_size_neg + pos_index + 1)]
    the_testing_neg = dataSet[(half_size_neg + pos_index + 1):]

    the_training.append(the_training_pos)
    the_training.append(the_training_neg)

    the_testing.append(the_testing_pos)
    the_testing.append(the_testing_neg)

    return the_training, the_testing


# delete the label from feature lists
def detachLabel(the_training):
    the_training_label = []
    for element in the_training:
        the_training_label.append(element[-1])
        element.pop()
    return the_training, the_training_label


def get_p_pos_and_neg(the_training):
    the_p_pos = len(the_training[0]) / float(len(the_training[1]))
    the_p_neg = 1 - the_p_pos

    return the_p_pos, the_p_neg


def probability_gaussian(one_testing_set, the_mean_list, the_std_list):
    the_features_probability = []
    square_root =  math.sqrt(2 * math.pi)

    for i in range(len(one_testing_set)):

        power_numerator = -math.pow((one_testing_set[i] - the_mean_list[i]),2)
        if the_std_list[i] == 0:
            the_std_list[i] = 0.01
        power_denominator = 2 * math.pow(the_std_list[i], 2)

        power = power_numerator / power_denominator

        denominator = square_root * the_std_list[i]
        numerator = math.exp(power)

        p_per_feature = numerator / denominator

        if numerator == 0:
            log_p_per_feature = power - math.log(denominator)
        else:
            log_p_per_feature = math.log(p_per_feature)
        the_features_probability.append(log_p_per_feature)

    return the_features_probability


def get_p_in_class(the_p, the_features_list):
    # because of probability is too small
    # I convert to log so that we can sum all these probabilities instead of product
    features_product = 0
    for i in range(len(the_features_list)):
        features_product += the_features_list[i]

    the_p_pos_log = math.log10(the_p)

    the_features_p_in_class = the_p_pos_log + features_product

    return the_features_p_in_class


def classifier(the_testing_pos_list, the_testing_neg_list, the_mean_list_pos, the_mean_list_neg,
                       the_std_list_pos, the_std_list_neg, the_p_pos, the_p_neg):
    count = 0
    the_tp = 0
    the_tn = 0
    the_fp = 0
    the_fn = 0
    for i in range(len(the_testing_pos_list)):
        features_p_pos_for_pos_list = probability_gaussian(the_testing_pos_list[i], the_mean_list_pos, the_std_list_pos)
        features_p_neg_for_pos_list = probability_gaussian(the_testing_pos_list[i], the_mean_list_neg, the_std_list_neg)

        features_p_in_pos_for_pos = get_p_in_class(the_p_pos, features_p_pos_for_pos_list)
        features_p_in_neg_for_pos = get_p_in_class(the_p_neg, features_p_neg_for_pos_list)

        max_value = max(features_p_in_pos_for_pos, features_p_in_neg_for_pos)
        if max_value == features_p_in_pos_for_pos:
            count += 1
            the_tp += 1
        else:
            the_fn += 1

    for j in range(len(the_testing_neg_list)):
        features_p_pos_for_neg_list = probability_gaussian(the_testing_neg_list[j], the_mean_list_pos, the_std_list_pos)
        features_p_neg_for_neg_list = probability_gaussian(the_testing_neg_list[j], the_mean_list_neg, the_std_list_neg)

        features_p_in_pos_for_neg = get_p_in_class(the_p_pos, features_p_pos_for_neg_list)
        features_p_in_neg_for_neg = get_p_in_class(the_p_neg, features_p_neg_for_neg_list)

        max_value = max(features_p_in_pos_for_neg, features_p_in_neg_for_neg)
        if max_value == features_p_in_neg_for_neg:
            count += 1
            the_tn += 1
        else:
            the_fp += 1

    return count, the_tp, the_tn, the_fp, the_fn


def print_confusion_matrix(the_tp, the_tn, the_fp, the_fn):
    print "\t\t\t\t" + "predicted_spam" + "\t" + "predicted_not_spam"
    print "actual_spam" + "\t\t\t", tp, "\t\t\t", fn
    print "actual_not_spam" + "\t\t", fp, "\t\t\t", tn



features, labels = LoadSpamData()
# features, labels = BalanceDataset(features, labels)
dataSet = attachLabel(features, labels)


# random.shuffle(dataSet)
training, testing = split(dataSet)
p_pos, p_neg = get_p_pos_and_neg(training)

training_pos, training_pos_label = detachLabel(training[0])
training_neg, training_neg_label = detachLabel(training[1])

testing_pos, testing_pos_label = detachLabel(testing[0])
testing_neg, testing_neg_label = detachLabel(testing[1])


training_pos = ConvertDataToArrays(training_pos)
training_neg = ConvertDataToArrays(training_neg)

testing_pos = ConvertDataToArrays(testing_pos)
testing_neg = ConvertDataToArrays(testing_neg)

mean_pos, std_pos = get_mean_and_std(training_pos)
mean_neg, std_neg = get_mean_and_std(training_neg)

correct, tp, tn, fp, fn = classifier(testing_pos, testing_neg, mean_pos, mean_neg, std_pos, std_neg, p_pos, p_neg)
# print correct
print "The accuracy is: ", correct/float(len(testing_pos) + len(testing_neg))
print "\n"
print_confusion_matrix(tp, tn, fp, fn)

print "\n"
print "precision is ", tp/float(tp+fp)
print "recall is ", tp/float(tp+fn)

# print len(mean_pos)
# print len(std_pos)


# print len(training_pos[0])

# means, std = get_mean_and_std(training)
# training = NormalizeFeatures(training, means, std)
# testing = NormalizeFeatures(testing, means, std)





