
# coding: utf-8

# In[124]:

# %load poi_id.py
#!/usr/bin/python

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm


# In[125]:

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# In[126]:

### Show the list of features and the number of features, one of them being a label : poi.
print data_dict['ALLEN PHILLIP K'].keys()
print '\nNumber of features including label poi: {0} '.format(len(data_dict['ALLEN PHILLIP K'].keys()))


# In[127]:

# number of poi in the data set
# I created a function to explore my data set
# in order to be able to do it any time later in the analysis process
# on other data sets

def explore_poi(data):

    ctr_poi = 0
    ctr_non_poi = 0
    poi_names = []
    for name, features in data.items():
        for feature, value in features.items():
            if feature == 'poi' and value == 1:
                poi_names.append(name)
                ctr_poi += 1
            elif feature == 'poi' and value == 0: 
                ctr_non_poi += 1
    fraction_poi = round(float(ctr_poi)/len(data), 2)            


    print np.array(sorted(poi_names))
    print "\nNumber of poi: {0}".format(ctr_poi)
    print "Number of non poi: {0}".format(ctr_non_poi)
    print "Number of data points in dataset: {0}".format(len(data))
    print "Fraction of poi in dataset: {0}".format(fraction_poi)


# In[128]:

explore_poi(data_dict)


# In[129]:

### I want to check data points with many NaN values
### it may be an indicator of an error or unnecessary data points.
### The function below returns a dictionary with names and number of NaN values

def explore_missing_features(data):
    missing_features_by_name = {}
    for name, features in data_dict.items():
        ctr = 0
        for feature, value in features.items():
            if value == 'NaN':
                ctr += 1
        missing_features_by_name[name] = ctr
    
    for name, value in missing_features_by_name.items():
            if value >= 17:
                print name, value
   


# In[130]:

explore_missing_features(data_dict)


# In[131]:

### Verify one data point to be sure that it does contain only NaN values
print data_dict['LOCKHART EUGENE E']


# In[132]:

### Now I will start to work on data_dict removing data, correcting them
### For this purpose I will use my_dataset

my_dataset = data_dict


# In[133]:

### I will remove data points 'LOCKHART EUGENE E' as it contains only NaN values
### and 'THE TRAVEL AGENCY IN THE PARK' as this is not a person 

### The function returns a new dictionary without data points that
### have to be removed.
### It takes as argument the key of the data point to remove and dictionary
### containing data points to remove.
### key is a list.
### The function returns a new dictionary

def remove_datapoint(keys_to_remove, datadict):
    new_datadict = {key:value for key, value in datadict.items() if key not in keys_to_remove}
    print 'The new dictionary contains {0} data points'.format(len(new_datadict))
    return new_datadict


# In[134]:

my_dataset = remove_datapoint(['THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E'], my_dataset)
explore_poi(my_dataset)


# In[135]:

### Task 2: Remove outliers
def outliers_detection(datadict, feature):
    i = int(round(len(datadict) * 0.05))

    feature_values = featureFormat(datadict, [feature], sort_keys = True)
    max_feature_values = sorted(feature_values, reverse = True)[0:i]
    min_feature_values = sorted(feature_values, reverse = False)[0:i]
    
    upper_outliers = {}
    lower_outliers = {}
    for name, features in datadict.items():
        for key, value in features.items():
            if key == feature and value != 'NaN' and value in max_feature_values:
                upper_outliers[name] = value
    for name, features in datadict.items():
        for key, value in features.items():
            if key == feature and value != 'NaN' and value in min_feature_values:
                lower_outliers[name] = value
    print sorted(upper_outliers.items(), key=lambda x:x[1])
    print sorted(lower_outliers.items(), key=lambda x:x[1])
 


# In[136]:


outliers_detection(my_dataset, 'total_payments')


# In[137]:

# The biggest outlier in total payments is the line TOTAL. To remove:
 
my_dataset = remove_datapoint(['TOTAL'], my_dataset)


# In[138]:

# After removal of the data point TOTAL I rerun statistics about dataset (number of data points etc) as well as 
# features selection
explore_poi(my_dataset)
outliers_detection(my_dataset, 'deferral_payments')


# In[139]:

# The lower outlier in deferral payments has the only negative value in this field.
# After consulting the pdf file I determined that this is a typo: deferral payment has been mixed up with deferred income.

print data_dict['BELFER ROBERT']

# Robert Belfer have been shifted probably when entered manually
# I will correct this.


# In[140]:

director_fees = 102500
expenses = 3285
restricted_stock = 44093
data_dict['BELFER ROBERT']['deferred_income'] = -director_fees
data_dict['BELFER ROBERT']['deferral_payments'] = 'NaN'
data_dict['BELFER ROBERT']['exercised_stock_options'] = 'NaN'
data_dict['BELFER ROBERT']['total_payments'] = expenses
data_dict['BELFER ROBERT']['expenses'] = expenses
data_dict['BELFER ROBERT']['director_fees'] = director_fees
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -restricted_stock
data_dict['BELFER ROBERT']['restricted_stock'] = restricted_stock
data_dict['BELFER ROBERT']['total_stock_value'] = 'NaN'


# In[141]:

outliers_detection(my_dataset, 'loan_advances')

# There are only 3 values different from NaN for this feature. Among these three persons there is one poi with very high value.
# That is the reason that this features has been selected by PercentileSelect based on F value.
# I will remove it from my list of features as we have not enough data for advanced loans.


# In[142]:

outliers_detection(my_dataset, 'restricted_stock')

# Here we can see that restricted stock value is negative which is not correct
# Data for 'BHATNAGAR SANJAY' have been shifted.
# I will correct this.


# In[143]:

restricted_stock = 2604490
expenses = 137864
exercised_stock_options = 15456290

data_dict['BHATNAGAR SANJAY']['restricted_stock'] = restricted_stock
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -restricted_stock
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = exercised_stock_options
data_dict['BHATNAGAR SANJAY']['director_fees'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['expenses'] = expenses
data_dict['BHATNAGAR SANJAY']['other'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['total_payments'] = expenses
data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = exercised_stock_options


# In[144]:

print data_dict['BHATNAGAR SANJAY']


# In[145]:

### I created list variable features_ini with initial features in order: 
### first financial features (in order as they appear on pdf document)
### then email features. This list excludes 'poi' (being a label) 
### and 'email_address' as it does not bring any information for my future analysis

features_ini = ['salary', 
                'bonus',
                'long_term_incentive',
                'deferred_income',
                'deferral_payments',
                'loan_advances',
                'other',
                'expenses',
                'director_fees',
                'total_payments',
                'exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred',
                'total_stock_value',
                'to_messages',
                'from_messages',
                'from_this_person_to_poi',
                'from_poi_to_this_person',
                'shared_receipt_with_poi']

### As I typed the list by hand, I now verify that all the elements are
### matching the elements in data_dict
for f in features_ini:
    if f not in data_dict['ALLEN PHILLIP K']:
        print 'excluded feature: {0}'.format(f)

for f in data_dict['ALLEN PHILLIP K'].keys():
    if f not in features_ini:
        print 'excluded feature: {0}'.format(f)
        
print 'Number of features in features_ini: {0}'.format(len(features_ini))

### I will now start with 19 features.


# In[146]:

if 'loan_advances' in features_ini:
    
    i = features_ini.index('loan_advances')
    features_ini = features_ini[0:i] + features_ini[i+1:]


# In[147]:

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### My comment: For purpose of feature selection I first created data using all features
### I removed email address from list of features at the beginning of the script. 
### After one iteration of selection I decided to remove loan_advances as well (too few data)
### This reduces my list to 18 features.

### The funciton below takes as argument a selection of features to evaluate (list),
### the data set (dictionary), and number n of features to select
### and returns 

def select_features(datadict, features_selection, n):
    
    features_list = ['poi'] + features_selection # I add 'poi' to my selection of features
    data = featureFormat(datadict, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    ### Now I will select features using KBest Selector
    ### I will use F-value to select features

    selector = SelectKBest(f_classif, k = n)
    selector.fit(features, labels)
    
    selected_indices = selector.get_support()
    unselected_indices = ~selector.get_support()

    selected_features = np.array(features_selection)[selected_indices].tolist()
    unselected_features = np.array(features_selection)[unselected_indices].tolist()
    
    all_scores = pd.DataFrame(selector.scores_, features_selection).sort_values(by = 0, ascending = False)
    
    print '\nSelected features {0}'.format(selected_features)
    print '\nUnselected features {0}'.format(unselected_features)
    return all_scores, selected_features


# In[148]:

all_scores, selected_features = select_features(my_dataset, features_ini, 12)
print all_scores


# In[149]:

### Task 3: Create new feature(s)

# Visualise some features in order to decide which new features to create

features_to_plot = ['poi', 'to_messages', 'shared_receipt_with_poi']
values_to_plot = featureFormat(my_dataset, features_to_plot, sort_keys = True)

plt.xlabel(features_to_plot[1])
plt.ylabel(features_to_plot[2])


for ii, pp in enumerate(values_to_plot):
    if values_to_plot[ii][0]:
        
        plt.scatter(values_to_plot[ii][1], 
                    values_to_plot[ii][2], color = 'r') 
    else:
        plt.scatter(values_to_plot[ii][1], 
                    values_to_plot[ii][2], color = 'b')
    
plt.xlim(0, )
plt.ylim(0, )
plt.show()

# Here I notice tht all the POI points are on the upper side.
# it seems that Poi have more shared receipt with poi for the same 
# number of messages received


# In[150]:

features_to_plot = ['poi', 'from_poi_to_this_person', 'shared_receipt_with_poi', 'to_messages']
values_to_plot = featureFormat(my_dataset, features_to_plot, sort_keys = True)

plt.xlabel('RATIO: {0} / {1}'.format(features_to_plot[1], features_to_plot[3]))
plt.ylabel('RATIO: {0} / {1}'.format(features_to_plot[2], features_to_plot[3]))

for ii, pp in enumerate(values_to_plot):
    if values_to_plot[ii][0]:
        
        plt.scatter(values_to_plot[ii][1]/float(values_to_plot[ii][3]), 
                    values_to_plot[ii][2]/float(values_to_plot[ii][3]), color = 'r') 
    else:
        plt.scatter(values_to_plot[ii][1]/float(values_to_plot[ii][3]), 
                    values_to_plot[ii][2]/float(values_to_plot[ii][3]), color = 'b')
plt.ylim()    
plt.show()

# On the plot below I notice that ratio on the y axis is less spread for POI
# than the ratio on x axis. So I will create the ratio od shared receipts with poi 
# to all received messages.


# In[151]:

# Visualise some features in order to decide which new features to create


features_to_plot = ['poi', 'salary', 'total_payments', 'total_stock_value']
values_to_plot = featureFormat(my_dataset, features_to_plot, sort_keys = True)

plt.xlabel(features_to_plot[1])
plt.ylabel('sum of {0} and {1} '.format(features_to_plot[2], features_to_plot[3]))


for ii, pp in enumerate(values_to_plot):
    if values_to_plot[ii][0]:
        
        plt.scatter(values_to_plot[ii][1], 
                    values_to_plot[ii][2] + values_to_plot[ii][3], color = 'r') 
    else:
        plt.scatter(values_to_plot[ii][1], 
                    values_to_plot[ii][2] + values_to_plot[ii][3], color = 'b')
    
plt.xlim()
plt.ylim()
plt.show()

# It is not clear if salary represents a lower or higher percentage in total_income for POI


# In[152]:

features_to_plot = ['poi', 'salary', 'total_payments', 'total_stock_value']
values_to_plot = featureFormat(my_dataset, features_to_plot, sort_keys = True)

plt.xlabel('RATIO: {0} / {1} and {2}'.format(features_to_plot[1], features_to_plot[2], features_to_plot[3]))
plt.ylabel(features_to_plot[1])

for ii, pp in enumerate(values_to_plot):
    if values_to_plot[ii][0]:
        
        plt.scatter(values_to_plot[ii][1]/float((values_to_plot[ii][2] + values_to_plot[ii][3])), 
                    values_to_plot[ii][1], color = 'r') 
    else:
        plt.scatter(values_to_plot[ii][1]/float((values_to_plot[ii][2] + values_to_plot[ii][3])), 
                    values_to_plot[ii][1], color = 'b')
    
plt.ylim()
plt.xlim()    
plt.show()

### Here we can see that ratio: salary to the sum of total_payments and total_stock_value
### does not increase above about 13-14% for POI
### At the same time salary of POI does not go below about 180000.
### I will add another feature: ratio_salary_to_total_income


# In[153]:

# Create new feature: percentage of shared_receipt_with_poi meassages in all received messages

feature_name = 'ratio_shared_with_poi_to_all_receipts'
feature1 = 'shared_receipt_with_poi'
feature2 = 'to_messages'

for name, features in my_dataset.items():
    for feature, value in my_dataset.items():
        if features[feature1] == 'NaN':
            features[feature_name] = 0
        else:
            features[feature_name] = round(float(features[feature1])/features[feature2], 2)


# In[154]:

# Create new fetaure: 
# percentage of salary in total income(the sum of total_payments and total_stock_value)

feature_name = 'ratio_salary_to_total_income'
feature1 = 'salary'
feature2 = 'total_payments'
feature3 = 'total_stock_value'

for name, features in my_dataset.items():
    for feature, value in features.items():
        if features[feature1] == 'NaN':
            features[feature_name] = 0
            
        elif features[feature3] == 'NaN':
            features[feature_name] = round(float(features[feature1])/
                                           features[feature2], 2)
        else:
            features[feature_name] = round(float(features[feature1])/
                                           (features[feature2] + features[feature3]), 2)


# In[155]:

# Create new feature: percentage of shared_receipt_with_poi meassages in all received messages

feature_name = 'ratio_to_from_poi'
feature1 = 'from_this_person_to_poi'
feature2 = 'from_poi_to_this_person'

for name, features in my_dataset.items():
    for feature, value in my_dataset.items():
        if features[feature1] == 'NaN' or features[feature2] == 'NaN' or features[feature2] == 0  :
            features[feature_name] = 0
        else:
            features[feature_name] = round(float(features[feature1])/features[feature2], 2)


# In[156]:

# Create new feature: 
# total_income (sum of total_payments and total_stock_value)

feature_name = 'total_income'
feature1 = 'total_payments'
feature2 = 'total_stock_value'

for name, features in my_dataset.items():
    for feature, value in features.items():
        if features[feature1] == 'NaN' and features[feature2] == 'NaN':
            features[feature_name] = 0
            
        elif features[feature1] == 'NaN':
            features[feature_name] = features[feature2]
        elif features[feature2] == 'NaN':
            features[feature_name] = features[feature1]
        else:
            features[feature_name] = features[feature1] + features[feature2]


# In[157]:

### Now I will test if newly created features score better

features_augmented = features_ini + ['total_income', 'ratio_salary_to_total_income', 
                                     'ratio_shared_with_poi_to_all_receipts', 
                                     'ratio_to_from_poi']
all_scores, selected_features = select_features(my_dataset, features_augmented, 12)
print all_scores


# In[159]:

### New features 'ratio_shared_with_poi_to_all_receipts' and 'total_income'
### got a high score
### On the less bright side, my another new feature 'ratio_salary_to_total_income'
### has the least score.
### I didn't expect that, but for the moment I will leave it this way.
### I will use features selected by KBest selector with F-value


# In[160]:

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# Gaussian Naive Bayes

def test_GaussianNB(dataset, selected_features):


    features_list = ['poi'] + selected_features # I add 'poi' to selected features
    data = featureFormat(dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    print precision_score(labels_test, pred, average = None)
    print recall_score(labels_test, pred, average = None)
    print accuracy_score(pred, labels_test)

    test_classifier(clf, dataset, features_list)


# In[161]:

# First test with KBest 12 features

dataset = my_dataset
all_scores, selected_features = select_features(dataset, features_augmented, 12)

test_GaussianNB(dataset, selected_features)


# In[162]:

# First test with KBest 3 features

dataset = my_dataset
all_scores, selected_features = select_features(dataset, features_augmented, 3)

test_GaussianNB(dataset, selected_features)


# In[163]:

# First test with selected by hand features
# Features that I consider conditionaly independent

dataset = my_dataset
#all_scores, selected_features = select_features(dataset, features_augmented, 3)
selected_features = ['total_income', 'bonus', 'ratio_shared_with_poi_to_all_receipts', 
                     'ratio_salary_to_total_income', 'ratio_to_from_poi', 
                     'from_poi_to_this_person']
test_GaussianNB(dataset, selected_features)


# In[164]:


# The function below is a copy of function from tester.py
# with one change: instead of dataset as argument
# I used labels and features
# I created this function to be able to test scaled features
# without need to update the dataset

PERF_FORMAT_STRING = "\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\tRecall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def my_test_classifier(clf, labels, features, feature_list, folds = 1000):
    
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."


# In[165]:

# SVC classifier 

# The function below tests the SVC classifier
# using as arguments dataset, selected features,
# the parameter C and the type of kernel

def test_SVC(dataset, selected_features, c_number, kernel_type):
    features_list = ['poi'] + selected_features # I add 'poi' to selected features
    data = featureFormat(dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    # Rescaling features as it is critical for SVC classifier
    # and features scales are very different
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)


    features_train, features_test, labels_train, labels_test = train_test_split(scaled_features, labels, test_size=0.3, random_state=42)

    print len(features_train)
    print len(features_test)

    clf = svm.SVC(C = c_number, kernel = kernel_type)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    print precision_score(labels_test, pred, average = None)
    print recall_score(labels_test, pred, average = None)

    my_test_classifier(clf, labels, scaled_features, features_list)


# In[166]:

# I will remove 'LAY KENNETH L' so that 
# rescaling is not affected by too high of a value

my_dataset_no_lay = remove_datapoint(['LAY KENNETH L'], my_dataset)
dataset = my_dataset_no_lay

# Kbest 12 features, C = 10 and kernel rbf
all_scores, selected_features = select_features(dataset, features_augmented, 12)
test_SVC(dataset, selected_features, 10, 'rbf')


# In[167]:

# Kbest 12 features, C = 100 and kernel rbf
all_scores, selected_features = select_features(dataset, features_augmented, 12)
test_SVC(dataset, selected_features, 100, 'rbf')


# In[168]:

# Kbest 12 features, C = 10 and kernel rbf
all_scores, selected_features = select_features(dataset, features_augmented, 12)
test_SVC(dataset, selected_features, 400, 'linear')


# In[169]:

# Decision Tree classifier

# This function is to test Decision Tree classifier
# arguments are dataset, selected_features, criterion ('gini' or 'entropy')
# and min sample split (split)

def test_DecisionTree(dataset, selected_features, crit, split):
    features_list = ['poi'] + selected_features
    data = featureFormat(dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split = split, criterion = crit)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print precision_score(labels_test, pred, average = None)
    print recall_score(labels_test, pred, average = None)
    
    importances = pd.DataFrame(selected_features, clf.feature_importances_)

    test_classifier(clf, dataset, features_list)

    print importances


# In[170]:

### First try: 9 features selected with KBest, criterion 'gini' and min smple split 10

dataset = my_dataset
all_scores, selected_features = select_features(dataset, features_augmented, 9)
#selected_features = ['bonus', 'deferred_income', 'exercised_stock_options', 'ratio_shared_with_poi_to_all_receipts']
test_DecisionTree(dataset, selected_features, 'gini', 10)

# The performance is pretty low


# In[171]:

# Now I will try with reduced features list: 
# only features with highest importance
# from previous test are used

dataset = my_dataset
#all_scores, selected_features = select_features(dataset, features_augmented, 9)
selected_features = ['bonus', 'deferred_income', 'exercised_stock_options', 'ratio_shared_with_poi_to_all_receipts']
test_DecisionTree(dataset, selected_features, 'gini', 10)

# The performance is much better


# In[172]:

# Now let's try a different min sample split (2 instead of 10)

dataset = my_dataset
#all_scores, selected_features = select_features(dataset, features_augmented, 9)
selected_features = ['bonus', 'deferred_income', 'exercised_stock_options', 'ratio_shared_with_poi_to_all_receipts']
test_DecisionTree(dataset, selected_features, 'gini', 2)

# The recall score is better with smaller min sample split
# and precision score lower
# I prefer this combination as recall score seems more important to me
# (see the questions document)


# In[173]:

# I would like to try bigger data set with criterion entropy (for information gain)
dataset = my_dataset
all_scores, selected_features = select_features(dataset, features_augmented, 9)
#selected_features = ['bonus', 'deferred_income', 'exercised_stock_options', 'ratio_shared_with_poi_to_all_receipts']
test_DecisionTree(dataset, selected_features, 'entropy', 2)

# When using a bigger features list I get better result with 
# criterion 'entropy' and smaller min sample split


# In[174]:

# Logistic Regression
# This function tests LR classifier with different features lists,
# class_weight (clw) and penalty (pen) parameters

def test_LogisticRegression(dataset, selected_features, pen, clw) :
    

    features_list = ['poi'] + selected_features
    data = featureFormat(dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)


    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(class_weight = clw, penalty = pen)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print precision_score(labels_test, pred, average = None)
    print recall_score(labels_test, pred, average = None)

    pred_scores = pd.DataFrame(labels_test, pred)

    test_classifier(clf, dataset, features_list)


# In[175]:

### I willl start with all features, and default parameters for
### penalty - 'l2'
### and class weight - None

k = len(features_augmented)
all_scores, selected_features = select_features(dataset, features_augmented, k)
 
dataset = my_dataset
test_LogisticRegression(dataset, selected_features, 'l2', None)


# In[176]:

### I willl try 12 features, penalty 'l2'
### and class weight None

k = 12
all_scores, selected_features = select_features(dataset, features_augmented, k)

dataset = my_dataset
test_LogisticRegression(dataset, selected_features, 'l2', None)


# In[177]:

### I willl try 6 features, penalty 'l2'
### and class weight None

k = 6
all_scores, selected_features = select_features(dataset, features_augmented, k)
   
dataset = my_dataset
test_LogisticRegression(dataset, selected_features, 'l2', None)


# In[178]:

### As I decreased the number of features used 
### the performance decreased too.
### I decide to keep all features 
### when training my Logistic Regression algotithm


# In[179]:

### I willl try all features, penalty 'l1'
### and class weight None

k = len(features_augmented)
all_scores, selected_features = select_features(dataset, features_augmented, k)

dataset = my_dataset
test_LogisticRegression(dataset, selected_features, 'l1', None)


# In[180]:

### I willl try 18 features just to confirm
### that lowering features number will still
### lower the performance  with another combination
### of paramteres: penalty 'l1'
### and class weight None

k = 18
all_scores, selected_features = select_features(dataset, features_augmented, k)

dataset = my_dataset
test_LogisticRegression(dataset, selected_features, 'l1', None)


# In[181]:

### I willl try all features, penalty 'l2'
### and class weight 'balanced'

k = len(features_augmented)
all_scores, selected_features = select_features(dataset, features_augmented, k)
    
dataset = my_dataset
test_LogisticRegression(dataset, selected_features, 'l2', 'balanced')


# In[182]:

### I willl try 12 features, penalty 'l2'
### and class weight 'balanced'

k = 12
all_scores, selected_features = select_features(dataset, features_augmented, k)

dataset = my_dataset
test_LogisticRegression(dataset, selected_features, 'l2', 'balanced')


# In[183]:

### I willl try all features, penalty 'l1'
### and class weight 'balanced'

k = len(features_augmented)
all_scores, selected_features = select_features(dataset, features_augmented, k)

dataset = my_dataset
test_LogisticRegression(dataset, selected_features, 'l1', 'balanced')


# In[184]:

### I willl try 12 features, penalty 'l1'
### and class weight 'balanced'

k = 12
all_scores, selected_features = select_features(dataset, features_augmented, k)

dataset = my_dataset
test_LogisticRegression(dataset, selected_features, 'l1', 'balanced')


# In[186]:

k = len(features_augmented)
all_scores, selected_features = select_features(dataset, features_augmented, k)


# In[187]:

clf = LogisticRegression(penalty = 'l1', class_weight = 'balanced')
features_list = ['poi'] + selected_features


# In[188]:

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# In[ ]:



