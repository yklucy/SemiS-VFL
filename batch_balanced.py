import numpy as np
from collections import Counter
import torch
from PIL import ImageFilter, Image
from torchvision import transforms
import random

#####################################################################
class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

#####################################################################
def balance_with_labels(trn_X_in, trn_y_in, num, args):
    print('args.dataset:{}'.format(args.dataset))
    
    # get the label distribution
    target = trn_y_in.view(-1)
    t_dict = Counter(target.numpy().tolist())

    # j is the key of t_dict
    X = None
    trn_y = None
    for j in t_dict.keys():
        if t_dict[j] > num:
            data_balancing_X = None
            trn_yy = None
            indices_value_under = (torch.nonzero(trn_y_in == j))[:,0]
            indices_under = indices_value_under

            data_balancing_X = trn_X_in[indices_under]
            # no undersampling, keeping the number
            trn_yy = torch.full((t_dict[j],),j)
        else:
            data_balancing_X = None
            trn_yy = None
            # oversampling
            indices_value_over = (torch.nonzero(trn_y_in == j))[:,0]
            data_balancing_X = trn_X_in[indices_value_over]
            trn_yy = torch.full((len(indices_value_over),),j)
            for i in range(num-t_dict[j]):    
                random_indices_over = torch.randperm(len(indices_value_over))[:1]
                index_over = indices_value_over[random_indices_over]
                tmp_X = trn_X_in[index_over]
                
                if args.dataset == 'EMNIST':
                    tmp_X = tmp_X.numpy().reshape(14, 28)
                    tmp_X = Image.fromarray(tmp_X,'L')
                    transform_img_EMNIST = transforms.Compose([
                                                        transforms.RandomResizedCrop((14,28), scale=(0.3, 1.)),
                                                        transforms.RandomApply([
                                                            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)
                                                        ], p=0.8),
                                                        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor()
                                                    ])
                    new_X = transform_img_EMNIST(tmp_X).reshape(1,-1)
                    
                if args.dataset == 'CIFAR10':
                    tmp_X = tmp_X.numpy().reshape(3, 16, 32)
                    tmp_X = tmp_X.transpose(1,2,0)
                    transform_img_CIFAR10 = transforms.Compose([
                                                            transforms.ToPILImage(mode='RGB'),
                                                            transforms.RandomResizedCrop((16,32), scale=(0.3, 1.)),
                                                            transforms.RandomApply([
                                                                transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)
                                                            ], p=0.8),
                                                            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ToTensor()
                                                    ])
                    new_X = transform_img_CIFAR10(tmp_X).reshape(1,-1)
                                        
                if args.dataset == 'NUSWIDE':
                    # Adding random noise as an example
                    tmp_X = np.array(tmp_X) + np.random.normal(0, 0.1, size=tmp_X.shape)  
                    new_X = torch.tensor(tmp_X, dtype=torch.float32)
                    
                new_trn_yy = torch.full((1,),j)
                data_balancing_X = torch.cat((data_balancing_X,new_X), dim=0)
                trn_yy = torch.cat((trn_yy, new_trn_yy),dim=0)

        # concat tensor with 10 classes
        if X == None:
            if data_balancing_X != None:
                X = data_balancing_X
                trn_y = trn_yy
        else:
            if data_balancing_X != None:
                X = torch.cat((X, data_balancing_X), dim=0)
                trn_y = torch.cat((trn_y, trn_yy), dim=0)

    return X, trn_y


