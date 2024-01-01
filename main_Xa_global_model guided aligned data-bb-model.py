import numpy as np
import random
import logging
import torch
import torch.nn as nn

from utils import AverageMeter, AverageMeterDict, perf_metrics, save_models, encrypt_with_iso, load_models
from exp_arguments import prepare_exp, save_exp_logs
from data.emnist import get_train_dataset_EMNIST, get_test_dataset_EMNIST
from data.cifar10 import get_test_dataset_CIFAR10, get_train_dataset_CIFAR10
from data.nuswide import get_train_dataset_NUSWIDE, get_test_dataset_NUSWIDE
from models.model_one import model_one
from models.global_model_Xa_Xb import Party_A_Classification, Party_B_Classification
from batch_balanced import balance_with_labels

def main():
    # read args
    args = prepare_exp()
    args.model_path = 'saved models'

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    #args.balanced = 0 -> imbalanced; default -> 1
    args.balanced = 0
    
    # save logs
    save_exp_logs(args, 'classic')

    num_classes = len(args.mul_classes)

    ######################################################
    # parameters related to dataset
    #args.dataset = 'CIFAR10'
    #args.dataset = 'EMNIST'
    print('args.dataset:{}'.format(args.dataset))


    if args.dataset == 'EMNIST':
        input_shape_Xa = (1, 14, 28)
        input_shape_Xb = (1, 14, 28)
        args, Xa_train_local, Xb_train_local, ya_train_local, Xa_aligned, Xb_aligned, y_aligned, Xa_unaligned, Xb_unaligned, ya_unaligned = get_train_dataset_EMNIST(args)
        args, Xa_test, Xb_test, y_test = get_test_dataset_EMNIST(args)
    
    if args.dataset == 'CIFAR10':
        input_shape_Xa = (3, 16, 32)
        input_shape_Xb = (3, 16, 32)
        args, Xa_train_local, Xb_train_local, ya_train_local, Xa_aligned, Xb_aligned, y_aligned, Xa_unaligned, Xb_unaligned, ya_unaligned = get_train_dataset_CIFAR10(args)
        args, Xa_test, Xb_test, y_test = get_test_dataset_CIFAR10(args)

    if args.dataset == 'NUSWIDE':
        args.models = 'mlp2'
        input_shape_Xa = 634
        input_shape_Xb = 1000
        args, Xa_train_local, Xb_train_local, ya_train_local, Xa_aligned, Xb_aligned, y_aligned, Xa_unaligned, Xb_unaligned, ya_unaligned = get_train_dataset_NUSWIDE(args)
        args, Xa_test, Xb_test, y_test = get_test_dataset_NUSWIDE(args)

    ########################################################
    # load data
    Xa_aligned_wrap = list(zip(torch.tensor(Xa_aligned, dtype=torch.float), y_aligned.astype('int64')))
    Xa_aligned_loader = torch.utils.data.DataLoader(Xa_aligned_wrap, batch_size=args.batch_size, pin_memory=False, drop_last=False)

    Xb_aligned_loader = torch.utils.data.DataLoader(torch.tensor(Xb_aligned, dtype=torch.float), batch_size=args.batch_size, pin_memory=False, drop_last=False)

    Xb_unaligned_loader = torch.utils.data.DataLoader(Xb_unaligned, batch_size=args.batch_size, pin_memory=False, drop_last=False)
    
    Xa_train_local_wrap = list(zip(torch.tensor(Xa_train_local, dtype=torch.float), ya_train_local.astype('int64')))
    Xa_train_local_loader = torch.utils.data.DataLoader(Xa_train_local_wrap, batch_size=args.batch_size, pin_memory=False, drop_last=False)


    # wrap and loader for test_dataset on Party A - Test_Xa
    Xa_test_wrap = list(zip(torch.tensor(Xa_test, dtype=torch.float), y_test.astype('int64')))
    Xa_test_loader = torch.utils.data.DataLoader(Xa_test_wrap, batch_size=args.batch_size, pin_memory=False, drop_last=False)

    ########################################################
    # load models
    if args.dataset == 'NUSWIDE':
        local_model_Xa = model_one(input_shape_Xa, args, dropout_rate=0.3)
    else:
        local_model_Xa = model_one(input_shape_Xa, args)
        
    local_model_Xa = local_model_Xa.to(args.device)
    global_model_Xa = Party_A_Classification(input_shape_Xa, args)
    global_model_Xa = global_model_Xa.to(args.device)
    global_model_Xb = Party_B_Classification(input_shape_Xb, args)
    global_model_Xb = global_model_Xb.to(args.device)

    ########################################################
    # load saved models
    # only load global_model_Xa
    if args.model_path !='':
        print('@@@@@@@@@@@@ load saved models: global models, local models @@@@@@@@@@@@@')
        print('args.model_path:{}'.format(args.model_path))
        global_model_Xa = load_models(global_model_Xa, args.models, 1, args, args.use_cross_model, args.use_local_model)
        global_model_Xb = load_models(global_model_Xb, args.models, 2, args, args.use_cross_model, args.use_local_model)
        local_model_Xa = load_models(local_model_Xa, args.models, '1-bb', args, use_cross_model=False, use_local_model=True)

    optimizer = torch.optim.SGD(local_model_Xa.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    ########################################################
    # train models
    best_acc_top1 = 0.
    best_f1_score_micro = 0.
    model_f1_score_micro = 0.
    flag_model = False

    ''' train and save models'''
    for epoch in range(args.epochs):
        print('hi.....I am training....{} times'.format(epoch))
        
        #####################################################################################
        # train()
        train_loss, train_acc, train_f1_score_micro, train_precision, train_recall, train_f1, train_roc_auc = train(num_classes, epoch, Xa_aligned_loader, Xb_aligned_loader, local_model_Xa, optimizer, args, global_model_Xa, global_model_Xb)

        # print avg train results
        for i in range(num_classes):
            if i in train_f1:
                logging.info(f'Class {i} - Precision: {train_precision[i]:.2f}, Recall: {train_recall[i]:.2f}, F1-Score: {train_f1[i]:.2f}, ROC_AUC: {train_roc_auc[i]:.2f}')
            else:
                logging.info(f'Class {i} - Precision: 0.00, Recall: 0.00, F1-Score: 0.00, ROC_AUC: 0.50')
 
        cur_step = (epoch+1) * len(Xa_train_local_loader)

        #####################################################################################
        # validation()
        test_losses, test_acc, test_f1_score_micro, test_precision, test_recall, test_f1, test_roc_auc = validation(num_classes, epoch, Xa_test_loader, None, local_model_Xa, args)

        for i in range(num_classes):
            if i in test_f1:
                logging.info(f'Class {i} - Precision: {test_precision[i]:.2f}, Recall: {test_recall[i]:.2f}, F1-Score: {test_f1[i]:.2f}, ROC_AUC: {test_roc_auc[i]:.2f}')
            else:
                logging.info(f'Class {i} - Precision: 0.00, Recall: 0.00, F1-Score: 0.00, ROC_AUC: 0.50')

        # save
        if test_acc > best_acc_top1:
            best_acc_top1 = test_acc

        if test_f1_score_micro > best_f1_score_micro:
            best_f1_score_micro = test_f1_score_micro

        if len(test_f1) == num_classes:
            if test_f1_score_micro > model_f1_score_micro:
                model_f1_score_micro = test_f1_score_micro
                """save models"""
                name_str =  args.models
                save_models(local_model_Xa, args.new_model_dir, args.dataset, name_str, '1-bb-tune', False, True)
                logging.info("***** model saved *****")    
                flag_model = True    
        
        scheduler.step()
        
        logging.info('best_acc_top1 %f', best_acc_top1)
        logging.info('best_f1_score_micro %f \n', best_f1_score_micro)
        
    if(flag_model == False):
        """save models"""
        name_str =  args.models
        save_models(local_model_Xa, args.new_model_dir, args.dataset, name_str, '1-bb-tune', False, True)
        logging.info("***** model saved *****")   
        flag_model = True

    print("main_Xa_global_model guided aligned data-bb-model.py")        
    print('local_model_Xa:{}'.format(local_model_Xa))

#####################################################################
def train(num_classes, epoch, Xa_aligned_loader, Xb_aligned_loader, local_model_Xa, optimizer, args, global_model_Xa, global_model_Xb):
    # train()
    acc_train = AverageMeter()
    losses = AverageMeter()
    f1_score_micro_train = AverageMeter()

    precision_train = AverageMeterDict(num_classes)
    recall_train = AverageMeterDict(num_classes)
    f1_train = AverageMeterDict(num_classes)
    roc_auc_train = AverageMeterDict(num_classes)    
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(args.device)

    cur_step = epoch * len(Xa_aligned_loader)

    local_model_Xa.train()

    Xa_aligned_iter = iter(Xa_aligned_loader)
    Xb_aligned_iter = iter(Xb_aligned_loader)

    for i in range(len(Xa_aligned_loader)):

        # data in party A - Xa
        Xa, y = next(Xa_aligned_iter)
        if args.balanced == 1:
            pass
        else:
            tran_Xa, trn_y = Xa.to(args.device), y
        
        target = trn_y.view(-1).long().to(args.device)
        N = target.size(0)            
        z_1 = None

        if args.dataset == 'EMNIST':
            tran_Xa = tran_Xa.reshape(-1, 1, 14, 28)
        if args.dataset == 'CIFAR10':
            tran_Xa = tran_Xa.reshape(-1, 3, 16, 32)

        Xb = next(Xb_aligned_iter)
        tran_Xb = Xb.to(args.device)
        if args.dataset == "EMNIST":
            tran_Xb = tran_Xb.reshape(-1, 1, 14, 28)
        if args.dataset == 'CIFAR10':
            tran_Xb = tran_Xb.reshape(-1, 3, 16, 32)

        z_0 = local_model_Xa(tran_Xa)

        with torch.no_grad():
            z_g_Xa = global_model_Xa(tran_Xa)
            z_g_Xb = global_model_Xb(tran_Xb)  
            out = torch.cat((z_g_Xa,z_g_Xb), dim=1)
            logits_g = global_model_Xa.global_model_classifier_head_cat(out) 
            loss_g = criterion(logits_g, target)

        logits = local_model_Xa.global_model_classifier_head_single(z_0) 
        if args.dataset == 'NUSWIDE':
            loss = criterion(logits, target) * 0.2 + loss_g * (1-0.2)
        if args.dataset == 'CIFAR10':
            loss = criterion(logits, target) * 0.2 + loss_g * (1-0.2)
        if args.dataset == 'EMNIST':
            loss = criterion(logits, target) * 0.2 + loss_g * (1-0.2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  

        acc, f1_score_micro, precision_dict, recall_dict, f1_dict, roc_auc_dict = perf_metrics(logits, target, args)
            
        losses.update(loss.item(), N)
        acc_train.update(acc.item(), N)
        f1_score_micro_train.update(f1_score_micro.item(), N)
                
        precision_train.update(precision_dict, N)
        recall_train.update(recall_dict, N)
        f1_train.update(f1_dict, N)
        roc_auc_train.update(roc_auc_dict, N)
        
        if i % args.report_freq == 0 or i == len(Xa_aligned_loader) - 1:
            logging.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "f1_score_micro  ({f1_score_micro.avg:.6f})".format(
                    epoch + 1, args.epochs, i, len(Xa_aligned_loader) - 1, losses=losses, f1_score_micro=f1_score_micro_train))

        cur_step += 1

    return losses.avg, acc_train.avg, f1_score_micro_train.avg, precision_train.avg, recall_train.avg, f1_train.avg, roc_auc_train.avg

#####################################################################
def validation(num_classes, epoch, Xa_test_loader, Xb_test_loader, local_model_Xa, args):
    # validation / test
    # validate()
    acc_test = AverageMeter()
    losses = AverageMeter()
    f1_score_micro_test = AverageMeter()

    precision_test = AverageMeterDict(num_classes)
    recall_test = AverageMeterDict(num_classes)
    f1_test = AverageMeterDict(num_classes)
    roc_auc_test = AverageMeterDict(num_classes)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(args.device)

    local_model_Xa.eval()

    with torch.no_grad():
        Xa_test_iter = iter(Xa_test_loader)
        
        for i in range(len(Xa_test_loader)):
            Xa, y = next(Xa_test_iter)
            val_Xa, val_y = Xa.to(args.device), y
        
            target = val_y.view(-1).long().to(args.device)
            N = target.size(0)
            if args.dataset == "EMNIST":
                val_Xa = val_Xa.reshape(-1, 1, 14, 28)
            if args.dataset == 'CIFAR10':
                val_Xa = val_Xa.reshape(-1, 3, 16, 32)

            z_0 = local_model_Xa(val_Xa)

            logits = local_model_Xa.global_model_classifier_head_single(z_0)

            loss = criterion(logits, target)

            acc, f1_score_micro, precision_dict, recall_dict, f1_dict, roc_auc_dict = perf_metrics(logits, target, args)

            losses.update(loss.item(), N)
            acc_test.update(acc.item(), N)
            f1_score_micro_test.update(f1_score_micro.item(), N)
            
            precision_test.update(precision_dict, N)
            recall_test.update(recall_dict, N)
            f1_test.update(f1_dict, N)
            roc_auc_test.update(roc_auc_dict, N)                

    logging.info(
        "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
        "f1_score_micro ({f1_score_micro.avg:.6f})".format(epoch + 1, args.epochs, i, len(Xa_test_loader) - 1, losses=losses, f1_score_micro=f1_score_micro_test))    

    return losses.avg, acc_test.avg, f1_score_micro_test.avg, precision_test.avg, recall_test.avg, f1_test.avg, roc_auc_test.avg


if __name__ == '__main__':
    main()