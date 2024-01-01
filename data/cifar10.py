import pickle
import os
import sys
import pandas as pd
import random
import logging

sys.path.append(os.getcwd())

from collections import Counter
from imblearn.datasets import make_imbalance

#####################################################################
def get_param_cifar10(args):
    # arg
    database_cifar10 = os.path.join(args.dataset_dir,"CIFAR10/")
    cifar10_path = "cifar-10-batches-py/"
    return database_cifar10, cifar10_path

#####################################################################
# file names
f_d_1 = "data_batch_1"
f_d_2 = "data_batch_2"
f_d_3 = "data_batch_3"
f_d_4 = "data_batch_4"
f_d_5 = "data_batch_5"
f_t = "test_batch"
f_meta = "batches.meta"

filename_list_train = [f_d_1, f_d_2, f_d_3, f_d_4, f_d_5]
filename_test = f_t

#####################################################################
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

#####################################################################
def unpickle_data(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

#####################################################################
# get_meta
def get_meta(database_cifar10, cifar10_path):
    # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    file_meta = os.path.join(database_cifar10, cifar10_path, "batches_meta.csv")
    if os.path.exists(file_meta):
        print('read batches_meta.csv')
        d_meta = pd.read_csv(file_meta)
    else:
        dict_meta = unpickle(f_meta)
        d_meta = pd.DataFrame.from_dict(dict_meta['label_names'])
        d_meta.to_csv(file_meta, index=False)
        
    return d_meta

#####################################################################
# get_data_fn_label(f_d_1)
def get_data_fn_label(database_cifar10, cifar10_path, filename):
    # load file
    filename_full = os.path.join(database_cifar10, cifar10_path, filename)
    data_batch = unpickle_data(filename_full)
    
    # get labels
    file_y = os.path.join(database_cifar10, cifar10_path, "_".join(["y", filename]) + ".csv")
    if os.path.exists(file_y):
        print('read y_{}.csv'.format(filename))
        y = pd.read_csv(file_y)
    else:
        y = pd.DataFrame.from_dict(data_batch[b'labels'])
        y.to_csv(file_y, index=False)
    
    counter_y = []
    for i in range(len(y)):
        counter_y.append(y.iloc[i,0])
    print('Counter y for {}:{}'.format(filename, Counter(counter_y)))
    
    # get data
    file_data = os.path.join(database_cifar10, cifar10_path, filename + ".csv")
    if os.path.exists(file_data):
        print('read {}.csv'.format(filename))
        d_df = pd.read_csv(file_data)
    else:
        d_df = pd.DataFrame.from_dict(data_batch[b'data'])
        d_df.to_csv(file_data, index=False)

    # get filename
    file_name = os.path.join(database_cifar10, cifar10_path, "_".join(["fn", filename]) + ".csv")
    if os.path.exists(file_name):
        d_fn = pd.read_csv(file_name)
    else:
        fn = data_batch[b'filenames']
        fn_de = []
        for i in range(len(fn)):
            fn_de.append(fn[i].decode("utf-8"))
        d_fn = pd.DataFrame(fn_de)
        d_fn.to_csv(file_name, index=False)
    
    return y.values, d_df.values, d_fn.values

#####################################################################
# train_dataset: 3072 = 1024+1024+1024 -> 32*32 + 32*32 + 32*32
# split one img into two party (32*16*2)

# input: pd.DataFrame
# output: type of dataset_Xa:<class 'numpy.ndarray'>
def split_dataset_twoparties(dataset):
    dataset = dataset.values
    dataset = dataset.reshape(len(dataset),3,32,32)
   
    # Transpose the whole data
    dataset = dataset.transpose(0,2,3,1)
    dataset_Xa = dataset[:,0:16,:,:]
    dataset_Xb = dataset[:,16:32,:,:]
    
    #####################################################
    # transfer data to the original format (len, 1536)
    dataset_Xa_transpose = dataset_Xa.transpose(0, 3, 1, 2)
    dataset_Xb_transpose = dataset_Xb.transpose(0, 3, 1, 2)

        
    split_Xa_tras_reshape = dataset_Xa_transpose.reshape(len(dataset_Xa_transpose),1536)
    split_Xb_tras_reshape = dataset_Xb_transpose.reshape(len(dataset_Xb_transpose),1536)   
    
    #return dataset_Xa, dataset_Xb
    return split_Xa_tras_reshape, split_Xb_tras_reshape

#####################################################################
def get_datasets(args):
    # file names
    f_d_1 = "data_batch_1"
    f_d_2 = "data_batch_2"
    f_d_3 = "data_batch_3"
    f_d_4 = "data_batch_4"
    f_d_5 = "data_batch_5"
    f_t = "test_batch"
    f_meta = "batches.meta"

    filename_list = [f_d_1, f_d_2, f_d_3, f_d_4, f_d_5]
    filename_test = f_t

    # loading ..... data
    print(' @@@@@@ loading....data.....@@@@@@')
    database_cifar10, cifar10_path = get_param_cifar10(args)
    args.mul_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    labels = []
    data = []
    
    # train
    if isinstance(filename_list, list):
        for i in range(len(filename_list)):
            label, d_df, d_fn = get_data_fn_label(database_cifar10, cifar10_path, filename_list[i])
            labels.append(pd.DataFrame(label))
            data.append(pd.DataFrame(d_df))
            
        y = pd.concat(labels, axis = 0)
        y = y.reset_index(drop=True)
        img_data = pd.concat(data, axis = 0)
        img_data = img_data.reset_index(drop=True)

        cifar10_y = []
        for i in range(y.shape[0]):
            cifar10_y.append(y.iloc[i,0])
        print('Train_dataset :{}'.format(Counter(cifar10_y)))
        
        # imbalance dataset for experimental requirements
        img_data_res, y_res = make_imbalance(img_data, cifar10_y, sampling_strategy={0:5000, 1:5000, 2:4500, 3:4500, 4:4000, 5:4000, 6:3500, 7:1000, 8:500, 9:250}, random_state=42)
        print('hi, successful imbalancing.....')
        print('Distribution after imbalancing:{}'.format(Counter(y_res)))   
        y = pd.DataFrame(y_res).values

    # test
    else:
        print(' this is test file .....')
        y, img_data, d_fn = get_data_fn_label(database_cifar10, cifar10_path, filename_test)
        img_data = pd.DataFrame(img_data)
        
        # Counter(y)
        cifar10_y = []
        for i in range(len(y)):
            cifar10_y.append(y[i][0])
        print('Test_dataset:{}'.format(Counter(cifar10_y)))        
        
        # imbalance dataset for experimental requirements
        img_data_res, y_res = make_imbalance(img_data, cifar10_y, sampling_strategy={0:1000, 1:1000, 2:800, 3:800, 4:500, 5:500, 6:250, 7:250, 8:100, 9:50}, random_state=42)
        print('hi, successful imbalancing.....')
        print('Distribution after imbalancing:{}'.format(Counter(y_res)))     
        print('y_res:{}'.format(y_res))
        y = pd.DataFrame(y_res).values    

    img_data = img_data_res
    
    Xa, Xb = split_dataset_twoparties(img_data)        
    print('type of Xa:{} and shape:{}'.format(type(Xa), Xa.shape))
    x = [Xa, Xb]
    print('{}: y - type:, shape{}'.format(type(y), y.shape))
    
    return args, Xa, Xb, y

#####################################################################
def get_test_dataset_CIFAR10(args):
    args, Xa_test, Xb_test, y_test = get_datasets(args)

    # shuffle
    n_test = len(Xa_test)
    test_indices = list(range(n_test))
    random.shuffle(test_indices)
    Xa_test = Xa_test[test_indices]
    Xb_test = Xb_test[test_indices]
    y_test = y_test[test_indices]
                        
    return args, Xa_test, Xb_test, y_test

#####################################################################
def get_train_dataset_CIFAR10(args):
    args, Xa_train, Xb_train, y_train = get_datasets(args)

    # indices of train and test datasets
    n_train = len(Xa_train)
    train_indices = list(range(n_train))

    # shuffle train samples
    random.shuffle(train_indices)
    
    logging.info("***** train data num： {}".format(len(train_indices)))
    
    """ step 2: get aligned labeled sampler (indices) , test sampler (indices) , train_local_sampler (indices) """
    train_aligned_labeled_num = int(n_train * args.aligned_samples_percent * args.labeled_samples_percent)
    train_aligned_labeled_indices = train_indices[:train_aligned_labeled_num]
    train_unaligned_labeled_indices = train_indices[train_aligned_labeled_num:-1]
    
    # all samples are local - n_train; train_indices
    logging.info("***** train_aligned_labeled_num:{}; train_local_num (all local train datasets):{}".format(train_aligned_labeled_num, n_train))

    # Xa and Xb local datasets
    Xa_train_local = Xa_train[train_indices]
    Xb_train_local = Xb_train[train_indices]
    ya_train_local = y_train[train_indices]

    # aligned datasets
    Xa_aligned = Xa_train[train_aligned_labeled_indices,:]
    Xb_aligned = Xb_train[train_aligned_labeled_indices,:]
    y_aligned = y_train[train_aligned_labeled_indices,:]

    # Xa_unaligned, ya_unaligned and Xb_unaligned
    Xa_unaligned = Xa_train[train_unaligned_labeled_indices,:]
    Xb_unaligned = Xb_train[train_unaligned_labeled_indices,:]
    ya_unaligned = y_train[train_unaligned_labeled_indices,:]

    return args, Xa_train_local, Xb_train_local, ya_train_local, Xa_aligned, Xb_aligned, y_aligned, Xa_unaligned, Xb_unaligned, ya_unaligned

