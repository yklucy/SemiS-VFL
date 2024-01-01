import numpy as np
import os
import torch

from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score

#####################################################################
class AverageMeter():
    #Computes and stores the average and current value
    def __init__(self):
        self.reset()

    def reset(self):
        #Reset all statistics
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # update statistics
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
#####################################################################
class AverageMeterDict():
    def __init__(self, len):
        self.len = len
        self.reset()
    
    def reset(self):
        self.sum = {}
        self.count = {}
        self.avg = {}

    # by batch                    
    def update(self, val_dict, n=1):
        self.val = {}
        for key, value in val_dict.items():
            self.val[key] = value
            if key in self.sum:
                self.sum[key] = self.sum[key] + self.val[key] * n
            else:
                self.sum[key] = self.val[key] * n
            if key in self.count:
                self.count[key] = self.count[key] + n
            else:
                self.count[key] = n
            self.avg[key] = self.sum[key] / self.count[key]

#####################################################################
def perf_metrics(logits, target, args):
    batch_size = target.size(0)

    _, pred = logits.topk(1, 1, True, True)
    pred_t = pred.t()
    correct = pred_t.eq(target.view(1, -1).expand_as(pred_t))
    correct_k = correct[0].reshape(-1).float().sum(0)
    acc = correct_k / batch_size

    #Precision, recall, f1-score and roc_auc
    pred_t_list = pred_t[0]
    target_au = np.array(target.tolist())
    pred_t_list_au = np.array(pred_t_list.tolist())
    # 10
    num_classes = len(args.mul_classes)
    # <= 10
    unique_classes = np.unique(target)
    
    # calculate allover f1-score
    f1_score_micro = f1_score(target_au, pred_t_list_au, average='micro', zero_division=0.0)

    precision, recall, f1, _ = precision_recall_fscore_support(target_au, pred_t_list_au, average=None, zero_division=0.0)
    
    # initial dict with 10
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    # store values as dict
    for i in range(len(precision)):
        if precision[i] != 0.0:
            precision_dict[i] = precision[i]
    for i in range(len(recall)):
        if recall[i] != 0.0:
            recall_dict[i] = recall[i]
    for i in range(len(f1)):
        if f1[i] != 0.0:
            f1_dict[i] = f1[i]

    # Calculate ROC AUC score for each class
    roc_auc_dict = {}
    for class_label in unique_classes:
        ovr_target = [1 if label == class_label else 0 for label in target]
        ovr_pred = [1 if pred == class_label else 0 for pred in pred_t_list]
        roc_auc = roc_auc_score(ovr_target, ovr_pred)
        
        # Store the ROC AUC score for the current class
        roc_auc_dict[class_label] = roc_auc

    return acc, f1_score_micro, precision_dict, recall_dict, f1_dict, roc_auc_dict

#####################################################################
def save_models(model, target_dir, dataset, name_str, i, use_cross_model, use_local_model):
    target_dir = os.path.join(target_dir, dataset)
    os.makedirs(target_dir, exist_ok=True)

    if use_cross_model and (use_local_model == False):
        torch.save(model.state_dict(),
                    os.path.join(target_dir, 'model_global-{}-{}.pth'.format(name_str, i)))
    if use_local_model and (use_cross_model == False):
        torch.save(model.state_dict(), os.path.join(target_dir, 'model_local-{}-{}.pth'.format(name_str, i)))

#####################################################################
def load_models(model, name_str, i, args, use_cross_model, use_local_model):
    # load saved global_model
    if use_cross_model:
        model.load_state_dict(torch.load('./{}/{}/model_global-'.format(args.new_model_dir, args.dataset) + name_str + '-{}.pth'.format(i), map_location=args.device))
        print('load model:{}'.format('./{}/{}/model_global-'.format(args.new_model_dir, args.dataset) + name_str + '-{}.pth'.format(i)))
    if use_local_model:
        model.load_state_dict(torch.load('./{}/{}/model_local-'.format(args.new_model_dir, args.dataset) + name_str + '-{}.pth'.format(i), map_location=args.device))
        print('load model:{}'.format('./{}/{}/model_local-'.format(args.new_model_dir, args.dataset) + name_str + '-{}.pth'.format(i)))
    return model

#####################################################################
# encrypt with iso
def encrypt_with_iso(g, ratio, th=5.0, device='cpu'):
    g = g.cpu()

    g_original_shape = g.shape
    g = g.view(g_original_shape[0], -1)

    g_norm = torch.norm(g, dim=1, keepdim=False)
    g_norm = g_norm.view(-1, 1)
    max_norm = torch.max(g_norm)
    gaussian_noise = torch.normal(size=g.shape, mean=0.0,
                                  std=1e-6+ratio * max_norm / torch.sqrt(torch.tensor(g.shape[1], dtype=torch.float32)))
    res = g + gaussian_noise
    res = res.view(g_original_shape).to(device)

    return res
