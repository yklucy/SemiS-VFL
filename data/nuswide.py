import os
import logging
import sys

import numpy as np
import pandas as pd
import copy
import random
sys.path.append(os.getcwd())

from sklearn.preprocessing._data import StandardScaler

#####################################################################
def get_param_nuswide(args):
    # arg
    dir_dataset_NUSWIDE = os.path.join(args.dataset_dir, "NUS-WIDE/")
    return dir_dataset_NUSWIDE

feature_path = "Low_Level_Features/"
tag_path = "NUS_WID_Tags/"
label_path = "Groundtruth/TrainTestLabels/"
concepts_path = "ConceptsList/"

#####################################################################
# 10 classes
#mul_classes = ['sky', 'clouds', 'person', 'water', 'animal', 'grass', 'buildings', 'window', 'plants', 'lake']
# example:
# get_concepts()
# selected_labels = get_concepts().values.ravel()
def get_concepts(dir_dataset_NUSWIDE):
    f_concepts = os.path.join(dir_dataset_NUSWIDE, concepts_path, "Concepts81.csv")
    if os.path.exists(f_concepts):
        print("read Concepts.csv")
        concepts_data = pd.read_csv(f_concepts)
    else:
        c_file = os.path.join(dir_dataset_NUSWIDE, concepts_path, "Concepts81.txt")
        concepts_data = pd.read_csv(c_file, header=None, sep=" ")
        concepts_data.dropna(axis=1, inplace=True)
        concepts_data.to_csv(f_concepts, index=False)
    return concepts_data

#####################################################################
# get labels from Groundtruth .txt files, 10 concepts/columns
# example:
# selected_labels =  ['sky', 'clouds', 'person', 'water', 'animal', 'grass', 'buildings', 'window', 'plants', 'lake'] 
# labels_train = get_labels(selected_labels, "Train")
# labels_train.shape[1]
# labels_test = get_labels(selected_labels, "Test")
# labels_test.shape[1]
def get_labels(selected_labels, dtype, dir_dataset_NUSWIDE):
    f_labels = os.path.join(dir_dataset_NUSWIDE, label_path, "_".join(["y_labels", dtype]) + ".csv")
    if os.path.exists(f_labels):
        print("read " + os.path.join("_".join(["y_labels", dtype]) + ".csv"))
        labels_data = pd.read_csv(f_labels)
    else:
        d_labels = []
        for label in selected_labels:
            l_file = os.path.join(dir_dataset_NUSWIDE, label_path, "_".join(["Labels", label, dtype]) + ".txt")
            df = pd.read_csv(l_file, header=None)
            df.columns = [label]
            d_labels.append(df)
        labels_data = pd.concat(d_labels, axis=1)
        labels_data.to_csv(f_labels, index=False)
    return labels_data

#####################################################################
def select_label(mul_classes, dtype, labels_data, dir_dataset_NUSWIDE):
    if len(mul_classes) > 1:
        selected = labels_data[labels_data.sum(axis=1) == 1]
    else:
        selected = labels_data
    print('selected_label.shape: {}'.format(selected.shape))
    selected.to_csv(os.path.join(dir_dataset_NUSWIDE, label_path, "_".join([dtype,"selected_lables"])+".csv"), index=False)

    return selected

#####################################################################
# get image low_level_features: CH, CM55, CORR, EDH, WT
# named XA -> 634 columns
# dtype: "Train", "Test"
# examples:
# features_train_selected = get_features("Train", selected)
# features_train_selected.shape[1]
# features_test_selected = get_features("Test", selected)
# features_test_selected.shape[1]
def get_features(dtype, selected, dir_dataset_NUSWIDE):
    f_features = os.path.join(dir_dataset_NUSWIDE, feature_path, "_".join(["X_features", dtype])+".csv")
    if os.path.exists(f_features):
        print("read " + os.path.join("_".join(["X_features", dtype])+".csv"))
        features_data_selected = pd.read_csv(f_features)
    else:
        df = []
        for filename in os.listdir(os.path.join(dir_dataset_NUSWIDE, feature_path)):
            if (filename.startswith(dtype)):
                #print(filename)
                f_file = os.path.join(dir_dataset_NUSWIDE, feature_path, filename)
                f_df = pd.read_csv(f_file, header=None, sep=" ")
                f_df.dropna(axis=1, inplace=True)
                #print(f_df.shape[1])
                df.append(f_df)
        features_data = pd.concat(df, axis=1)
        
        # selected samples
        features_data_selected = features_data.loc[selected.index]
        features_data_selected.to_csv(f_features, index=False)
    return features_data_selected

#####################################################################
# get tags
# named XB -> 1000 columns
# dtype: "Train", "Test"
# ttype: "Tags1k", "Tags81"
# ftype: ".dat", ".txt"
# t_sep: "\t", " "
# examples:
# tags_train = get_tags("Train", "Tags1k", ".dat", " ", selected_tags_test)
# tags_train.shape[1]
# tags_test = get_tags("Test", "Tags1k", ".dat", " ", selected_tags_test)
# tags_test.shape[1]
def get_tags(dtype, ttype, ftype, t_sep, selected, dir_dataset_NUSWIDE):
    f_tags = os.path.join(dir_dataset_NUSWIDE, tag_path, "_".join(["X", ttype, dtype])+".csv")
    if os.path.exists(f_tags):
        print("read " + os.path.join("_".join(["X_tags", dtype])+".csv"))
        tags_data_selected = pd.read_csv(f_tags)
    else:
        t_file = os.path.join(dir_dataset_NUSWIDE, tag_path, "_".join([dtype, ttype])+ftype)
        tags_data = pd.read_csv(t_file, header=None, sep=t_sep)
        tags_data.dropna(axis=1, inplace=True)
        
        tags_data_selected = tags_data.loc[selected.index]
        
        tags_data_selected.to_csv(f_tags, index=False)
    return tags_data_selected

#####################################################################
# transfer y from OneHot to Category -> one column (y) with 10 classes
def label_trans(y):
    y_ = []
    pos_count = 0
    neg_count = 0
    count = {}
    # y is an array
    for i in range(y.shape[0]):
        # get the index of the nonzero label
        # transform OneHot to category - > one column (y) with 10 classes
        label = np.nonzero(y[i,:])[0][0]
        y_.append(label)
        if label not in count:
            count[label] = 1
        else:
            count[label] = count[label] + 1
    logging.info("***** Counter:{}".format(count))
    
    y = np.expand_dims(y_, axis=1)
    
    return y

#####################################################################
def get_datasets(dtype, args):
    # loading ..... data
    print(' @@@@@@ loading....data.....@@@@@@')
    dir_dataset_NUSWIDE = get_param_nuswide(args)
    mul_classes = args.mul_classes
    # labels
    labels = get_labels(mul_classes, dtype, dir_dataset_NUSWIDE)
    
    # selected labels
    selected_label = select_label(mul_classes, dtype, labels, dir_dataset_NUSWIDE)

    # selected features - Xa [634]
    selected_features = get_features(dtype, selected_label, dir_dataset_NUSWIDE)

    # selected tags - Xb [1000]
    selected_tags = get_tags(dtype, "Tags1k", ".dat", "\t", selected_label, dir_dataset_NUSWIDE)
    
    print(' @@@@@@ StandardScaler....data.....@@@@@@')
    # StandardScaler data
    data_scaler_model = StandardScaler()
    Xa = data_scaler_model.fit_transform(selected_features.values)
    Xb = data_scaler_model.fit_transform(selected_tags.values)
    x = [Xa, Xb]
    y = label_trans(selected_label.values)
   
    return args, Xa, Xb, y

#####################################################################
#Xa_test, Xb_test, y_test = get_datasets('test', args)
#Xa_test shape:(38955, 634) and type<class 'numpy.ndarray'>
#Xb_test shape:(38955, 1000) and type<class 'numpy.ndarray'>
#y_test shape:(38955, 1) and type<class 'numpy.ndarray'>
def get_test_dataset_NUSWIDE(args):
    args, Xa_test, Xb_test, y_test = get_datasets('Test', args)
    return args, Xa_test, Xb_test, y_test

#####################################################################
def get_train_dataset_NUSWIDE(args):
    args, Xa_train, Xb_train, y_train = get_datasets('Train', args)

    # indices of train and test datasets
    n_train = len(Xa_train)
    train_indices = list(range(n_train))

    # shuffle train samples
    random.shuffle(train_indices)
    
    logging.info("***** train data num： {}".format(len(train_indices)))

    """ step 2: get aligned labeled sampler (indices) , test sampler (indices) , train_local_sampler (indices) """
    # aligned samples - default value is 20% of all samples
    # labeled samples - default value is 100% of aligned samples
    train_aligned_labeled_num = int(n_train * args.aligned_samples_percent * args.labeled_samples_percent)
    train_aligned_labeled_indices = train_indices[:train_aligned_labeled_num]
    train_unaligned_labeled_indices = train_indices[train_aligned_labeled_num:-1]
    
    # all samples are local - n_train; train_indices
    logging.info("***** train_aligned_labeled_num:{}; train_local_num (all local train datasets):{}".format(train_aligned_labeled_num, n_train))

    # Xa and Xb local datasets
    Xa_train_local = Xa_train
    Xb_train_local = Xb_train
    ya_train_local = y_train

    # aligned datasets
    Xa_aligned = Xa_train[train_aligned_labeled_indices,:]
    Xb_aligned = Xb_train[train_aligned_labeled_indices,:]
    y_aligned = y_train[train_aligned_labeled_indices,:]

    # Xa_unaligned, ya_unaligned and Xb_unaligned
    Xa_unaligned = Xa_train[train_unaligned_labeled_indices,:]
    Xb_unaligned = Xb_train[train_unaligned_labeled_indices,:]
    ya_unaligned = y_train[train_unaligned_labeled_indices,:]

    return args, Xa_train_local, Xb_train_local, ya_train_local, Xa_aligned, Xb_aligned, y_aligned, Xa_unaligned, Xb_unaligned, ya_unaligned

#####################################################################
def data_aug(data):
    len_data = len(data)

    percent = 0.3

    # data_min shape:(100,); data_max shape:(100,)
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    
    #shape:(150, 100)
    data_q = copy.deepcopy(data)
 
    for i in range(len_data):
        masked_dim = int(data.shape[-1]*percent)

        # ind = np.random.choice(634, 190, replace=False) - > random choice a column number between 0 to 634; choice 190 times
        # len(ind) -> 190
        # masked_index:[97 50 96 53 11 43 54 76 70 79 99 67 25 81 60 44 33 36 13 95 88 49 58 90 75 45 19 80 63 87]
        masked_index = np.random.choice(data.shape[-1], masked_dim, replace=False)

        # masked_value shape:(30,)
        masked_value = np.random.uniform(data_min[masked_index], data_max[masked_index])
        data_q[i][masked_index] = masked_value

    data_a_q = np.concatenate((data, data_q),axis=1)
    return data_a_q