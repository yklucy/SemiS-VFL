# SemiS-VFL
A Semi-supervised Machine Learning Framework for Vertical Federated Learning

## Stage 1: main_vfl_global_model_Xa_Xb.py
python 'main_vfl_global_model_Xa_Xb_NUSWIDE.py' --dataset 'NUSWIDE' --epochs 30

python 'main_vfl_global_model_Xa_Xb_NUSWIDE.py' --dataset 'CIFAR10' --epochs 30

python 'main_vfl_global_model_Xa_Xb_NUSWIDE.py' --dataset 'EMNIST' --epochs 30

## 2. Stage 2: local model Xa
### Step 1: load global_model_Xa and tune it with all local samples and labels
python 'main_Xa_local_model_load_globalmodelA.py' --dataset 'NUSWIDE' --epochs 30

python 'main_Xa_local_model_load_globalmodelA.py' --dataset 'CIFAR10' --epochs 30

python 'main_Xa_local_model_load_globalmodelA.py' --dataset 'EMNIST' --epochs 30

### Step 2: load global_model_Xa and tune it with batch-balancing technique
python 'main_Xa_local_model_bb_load_globalmodelA.py' --dataset 'NUSWIDE' --epochs 30

python 'main_Xa_local_model_bb_load_globalmodelA.py' --dataset 'CIFAR10' --epochs 30

python 'main_Xa_local_model_bb_load_globalmodelA.py' --dataset 'EMNIST' --epochs 30
### Step 3: tune local_model_Xa guided by global models
python 'main_Xa_global_model guided aligned data-bb-model.py' --dataset 'NUSWIDE' --epochs 30

python 'main_Xa_global_model guided aligned data-bb-model.py' --dataset 'CIFAR10' --epochs 30

python 'main_Xa_global_model guided aligned data-bb-model.py' --dataset 'EMNIST' --epochs 30

## 3. Stage 3: local model Xb
### Step 1: train local_model_Xb with aligned samples and labels
python 'main_Xb_local_model.py' --dataset 'NUSWIDE' --epochs 30

python 'main_Xb_local_model.py' --dataset 'CIFAR10' --epochs 30

python 'main_Xb_local_model.py' --dataset 'EMNIST' --epochs 30
### Step 2: load local_model_Xb and tune it with batch-balancing technique
python 'main_Xb_local_model_bb.py' --dataset 'NUSWIDE' --epochs 30

python 'main_Xb_local_model_bb.py' --dataset 'CIFAR10' --epochs 30

python 'main_Xb_local_model_bb.py' --dataset 'EMNIST' --epochs 30
### Step 3: tune local_model_Xb guided by global models
python 'main_Xb_global_model guided aligned data-bb-model.py' --dataset 'NUSWIDE' --epochs 30

python 'main_Xb_global_model guided aligned data-bb-model.py' --dataset 'CIFAR10' --epochs 30

python 'main_Xb_global_model guided aligned data-bb-model.py' --dataset 'EMNIST' --epochs 30
### Step 4: load local_model_Xb_bbtune and use it to train a semi-supervised learning
python 'main_Xb_pseudo labels_semi-supervised local model-bbtune.py' --dataset 'NUSWIDE' --epochs 30

python 'main_Xb_pseudo labels_semi-supervised local model-bbtune.py' --dataset 'CIFAR10' --epochs 30

python 'main_Xb_pseudo labels_semi-supervised local model-bbtune.py' --dataset 'EMNIST' --epochs 30

## 4. Stage 4: tune global models
python 'main_vfl_global_model_Xa_Xb_tuning.py' --dataset 'NUSWIDE' --epochs 30

python 'main_vfl_global_model_Xa_Xb_tuning.py' --dataset 'CIFAR10' --epochs 30

python 'main_vfl_global_model_Xa_Xb_tuning.py' --dataset 'EMNIST' --epochs 30
