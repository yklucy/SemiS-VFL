import numpy as np
from collections import Counter
import random
import logging
import torch
import torch.nn as nn

from utils import AverageMeter, AverageMeterDict, perf_metrics, save_models, encrypt_with_iso
from exp_arguments import prepare_exp, save_exp_logs
from data.emnist import get_train_dataset_EMNIST, get_test_dataset_EMNIST
from data.cifar10 import get_test_dataset_CIFAR10, get_train_dataset_CIFAR10
from data.nuswide import get_train_dataset_NUSWIDE, get_test_dataset_NUSWIDE

from models.global_model_Xa_Xb import Party_A_Classification, Party_B_Classification

#####################################################################
def main():
    # read args
    args = prepare_exp()

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    #args.balanced = 0 -> imbalanced; default -> 1
    args.balanced = 0
    
    # save logs
    save_exp_logs(args, 'classic')

    num_classes = len(args.mul_classes)

    use_cross_model = True
    use_local_model = False

    ######################################################
    # parameters related to dataset
    #args.dataset = 'CIFAR10'
    #args.dataset = 'EMNIST'
    print('args.dataset:{}'.format(args.dataset))
    ######################################################

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

    # wrap and loader for train_dataset on Party A - Train_Xa
    Xa_aligned_wrap = list(zip(torch.tensor(Xa_aligned, dtype=torch.float), y_aligned.astype('int64')))
    Xa_aligned_loader = torch.utils.data.DataLoader(Xa_aligned_wrap, batch_size=args.batch_size, pin_memory=False, drop_last=False)

    # loader for train_dataset on Party B - Train_Xb
    Xb_aligned_wrap = list(zip(torch.tensor(Xb_aligned, dtype=torch.float), y_aligned.astype('int64')))
    Xb_aligned_loader = torch.utils.data.DataLoader(Xb_aligned_wrap, batch_size=args.batch_size, pin_memory=False, drop_last=False)

    Xb_unaligned_loader = torch.utils.data.DataLoader(Xb_unaligned, batch_size=args.batch_size, pin_memory=False, drop_last=False)

    # wrap and loader for test_dataset on Party A - Test_Xa
    Xa_test_wrap = list(zip(torch.tensor(Xa_test, dtype=torch.float), y_test.astype('int64')))
    Xa_test_loader = torch.utils.data.DataLoader(Xa_test_wrap, batch_size=args.batch_size, pin_memory=False, drop_last=False)

    # loader for test_dataset on Party B - Test_Xb
    Xb_test_loader = torch.utils.data.DataLoader(torch.tensor(Xb_test, dtype=torch.float), batch_size=args.batch_size, pin_memory=False, drop_last=False)

    ########################################################
    # load models

    # models for Xa
    if args.dataset == 'NUSWIDE':
        model_Xa = Party_A_Classification(input_shape_Xa, args)
        model_Xa = model_Xa.to(args.device)

        # models for Xb
        model_Xb = Party_B_Classification(input_shape_Xb, args)
        model_Xb = model_Xb.to(args.device)
    elif args.dataset == 'CIFAR10':
        model_Xa = Party_A_Classification(input_shape_Xa, args)
        model_Xa = model_Xa.to(args.device)

        # models for Xb
        model_Xb = Party_B_Classification(input_shape_Xb, args)
        model_Xb = model_Xb.to(args.device)
    else:
        # EMNIST
        model_Xa = Party_A_Classification(input_shape_Xa, args)
        model_Xa = model_Xa.to(args.device)

        # models for Xb
        model_Xb = Party_B_Classification(input_shape_Xb, args)
        model_Xb = model_Xb.to(args.device)

    ########################################################
    # criterion, optimizer, scheduler

    # criterion
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(args.device)

    # weights optimizer
    optimizer_Xa = torch.optim.SGD(model_Xa.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler_Xa = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_Xa, args.epochs)

    optimizer_Xb = torch.optim.SGD(model_Xb.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler_Xb = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_Xb, args.epochs)

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
        train_loss, train_acc, train_f1_score_micro, train_precision, train_recall, train_f1, train_roc_auc = train(num_classes, epoch, Xa_aligned_loader, Xb_aligned_loader, optimizer_Xa, optimizer_Xb, criterion, model_Xa, model_Xb, args)
        
        # print avg train results
        for i in range(num_classes):
            if i in train_f1:
                logging.info(f'Class {i} - Precision: {train_precision[i]:.2f}, Recall: {train_recall[i]:.2f}, F1-Score: {train_f1[i]:.2f}, ROC_AUC: {train_roc_auc[i]:.2f}')
            else:
                logging.info(f'Class {i} - Precision: 0.00, Recall: 0.00, F1-Score: 0.00, ROC_AUC: 0.50')
 
        cur_step = (epoch+1) * len(Xa_aligned_loader)
        
        #####################################################################################
        # validation()
        test_losses, test_acc, test_f1_score_micro, test_precision, test_recall, test_f1, test_roc_auc = validation(num_classes, epoch, Xa_test_loader, Xb_test_loader, criterion, model_Xa, model_Xb, args)

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

                ########################################################
                # save models
                """save models"""
                name_str =  args.models
                save_models(model_Xa, args.new_model_dir, args.dataset, name_str, 1, use_cross_model, use_local_model)
                save_models(model_Xb, args.new_model_dir, args.dataset, name_str, 2, use_cross_model, use_local_model)
                logging.info("***** model saved *****")
                flag_model = True
        

        logging.info('best_accuracy: %f', best_acc_top1)
        logging.info('best_f1_score_micro %f \n', best_f1_score_micro)
        
        scheduler_Xa.step()
        scheduler_Xb.step()

    if(flag_model == False):
        # save models
        """save models"""
        name_str =  args.models
        save_models(model_Xa, args.new_model_dir, args.dataset, name_str, 1, use_cross_model, use_local_model)
        save_models(model_Xb, args.new_model_dir, args.dataset, name_str, 2, use_cross_model, use_local_model)
        logging.info("***** model saved *****")
        flag_model = True
        
    print('main_vfl_global_model_Xa_Xb.py')
    print('model_Xa:{}'.format(model_Xa))
    print('model_Xb:{}'.format(model_Xb))

########################################################

def train(num_classes, epoch, Xa_aligned_loader, Xb_aligned_loader, optimizer_Xa, optimizer_Xb, criterion, model_Xa, model_Xb, args):
    #################################################################
    # train()
    acc_train = AverageMeter()
    f1_score_micro_train = AverageMeter()
    losses = AverageMeter()
    precision_train = AverageMeterDict(num_classes)
    recall_train = AverageMeterDict(num_classes)
    f1_train = AverageMeterDict(num_classes)
    roc_auc_train = AverageMeterDict(num_classes)        

    cur_step = epoch * len(Xa_aligned_loader)
    cur_lr = optimizer_Xa.param_groups[0]['lr']
    logging.info("Epoch {}, cur_step {}, LR {}".format(epoch, cur_step, cur_lr))

    model_Xa.train()
    model_Xb.train()

    Xa_aligned_iter = iter(Xa_aligned_loader)
    Xb_aligned_iter = iter(Xb_aligned_loader)

    for i in range(len(Xa_aligned_loader)):
        Xa, y_Xa = next(Xa_aligned_iter)
        tran_Xa = Xa.to(args.device)
        target_Xa = y_Xa.view(-1).long().to(args.device)
        N = target_Xa.size(0)            
        z_1 = None

        if args.dataset == 'EMNIST':
            tran_Xa = tran_Xa.reshape(-1, 1, 14, 28)
        if args.dataset == 'CIFAR10':
            tran_Xa = tran_Xa.reshape(-1, 3, 16, 32)
        z_0 = model_Xa(tran_Xa)

        Xb, y_Xb = next(Xb_aligned_iter)
        tran_Xb = Xb.to(args.device)

        if args.dataset == 'EMNIST':
            tran_Xb = tran_Xb.reshape(-1, 1, 14, 28)
        if args.dataset == 'CIFAR10':
            tran_Xb = tran_Xb.reshape(-1, 3, 16, 32)
        z_rest = model_Xb(tran_Xb)

        z_1 = z_rest.detach().clone()
        z_1 = torch.autograd.Variable(z_1, requires_grad=True).to(args.device)

        out = torch.cat((z_0, z_1), dim=1)
        logits = model_Xa.global_model_classifier_head_cat(out)
        loss = criterion(logits, target_Xa)

        #####################################################################
        # manually calculate the gradients for model_Xb.parameters() using torch.autograd.grad
        z_gradients = torch.autograd.grad(loss, z_1, retain_graph=True)
        
        # default cls_iso_sigma = 0
        if args.cls_iso_sigma > 0:
            z_gradients = encrypt_with_iso(z_gradients[0], args.cls_iso_sigma, args.cls_iso_threshold, args.device)

        weights_gradients = torch.autograd.grad(z_rest, model_Xb.parameters(), grad_outputs=z_gradients, retain_graph=True, allow_unused=True)
        optimizer_Xa.zero_grad()
        loss.backward()
        # default grad_clip = 5
        if args.grad_clip > 0:
            # calculate the norm over all gradients together, input Tensor, return vector
            nn.utils.clip_grad_norm_(model_Xa.parameters(), args.grad_clip)

        optimizer_Xa.step()

        # optimizer - Xb
        optimizer_Xb.zero_grad()

        for w, g in zip(model_Xb.parameters(), weights_gradients):
            if w.requires_grad:
                w.grad = g.detach()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model_Xb.parameters(), args.grad_clip)

        optimizer_Xb.step()

        acc, f1_score_micro, precision_dict, recall_dict, f1_dict, roc_auc_dict = perf_metrics(logits, target_Xa, args)
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
                "f1_score_micro ({f1_score_micro.avg:.6f})".format(
                    epoch + 1, args.epochs, i, len(Xa_aligned_loader) - 1, losses=losses, f1_score_micro=f1_score_micro_train))

        cur_step += 1

    return losses.avg, acc_train.avg, f1_score_micro_train.avg, precision_train.avg, recall_train.avg, f1_train.avg, roc_auc_train.avg


#####################################################################
def validation(num_classes, epoch, Xa_test_loader, Xb_test_loader, criterion, model_Xa, model_Xb, args):
    #####################################################################################
    # validation / test
    # validate()
    acc_test = AverageMeter()
    f1_score_micro_test = AverageMeter()
    losses = AverageMeter()
    precision_test = AverageMeterDict(num_classes)
    recall_test = AverageMeterDict(num_classes)
    f1_test = AverageMeterDict(num_classes)
    roc_auc_test = AverageMeterDict(num_classes)

    model_Xa.eval()
    model_Xb.eval()


    with torch.no_grad():
        Xa_test_iter = iter(Xa_test_loader)
        Xb_test_iter = iter(Xb_test_loader)

        for i in range(len(Xa_test_loader)):
            Xa, y = next(Xa_test_iter)
            val_Xa, val_y = Xa.to(args.device), y          
            target = val_y.view(-1).long().to(args.device)
            N = target.size(0)
            z_1 = None
            
            if args.dataset == 'EMNIST':
                val_Xa = val_Xa.reshape(-1, 1, 14, 28)
            if args.dataset == 'CIFAR10':
                val_Xa = val_Xa.reshape(-1, 3, 16, 32)
            z_0 = model_Xa(val_Xa)

            Xb = next(Xb_test_iter)
            val_Xb = Xb.to(args.device)
            if args.dataset == 'EMNIST':
                val_Xb = val_Xb.reshape(-1, 1, 14, 28)
            if args.dataset == 'CIFAR10':
                val_Xb = val_Xb.reshape(-1, 3, 16, 32)
            z_rest = model_Xb(val_Xb)
            z_1 = z_rest.detach().clone()
            z_1 = torch.autograd.Variable(z_1, requires_grad=True).to(args.device)
            
            out = torch.cat((z_0, z_1), dim=1)
            logits = model_Xa.global_model_classifier_head_cat(out)

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