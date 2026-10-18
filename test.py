import argparse
import os
import warnings
import sys
import random
import numpy as np
import scipy
import torch
from sklearn import metrics

from evaluation_metrics import EvaluationMetrics
import copy

from loadMatData import  load_data
import scipy  as sp
from label_utils import reassign_labels, special_train_test_split
from data import Multi_view_data, generate_partition
from models import SparseInferenceModule
np.set_printoptions(threshold=np.inf)
import torch.nn.functional as F

def count_each_class_num(labels):
    '''
        Count the number of samples in each class
    '''
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict
def generate_partition(labels,ind, ratio=0.1):
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num = {}  ## number of labeled samples for each class
    total_num = round(ratio * len(labels))
    for label in each_class_num.keys():
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1)  # min is 1

    # index of labeled and unlabeled samples
    p_labeled = []
    p_unlabeled = []
    for idx, label in enumerate(labels):
        if (labeled_each_class_num[label] > 0):
            labeled_each_class_num[label] -= 1
            p_labeled.append(ind[idx])
            total_num -= 1
        else:
            p_unlabeled.append(ind[idx])
    return p_labeled, p_unlabeled

def mixup_data(x, y, alpha=0.3, use_cuda=True):
    lam = np.random.beta(alpha, alpha)
    batch_size = y.size()[0]
    sample=0
    trynum=0
    while(sample<batch_size/2 and trynum<10):
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        #   剔除同类对
        index_dc=torch.nonzero(y[index] != y).squeeze()
        sample=len(index_dc)
        trynum+=1
    mixed_x={}
    mixed_x[0] =lam *  x[0][index_dc] + (1 - lam) * x[0][index][index_dc]
    for i in range(1,len(x)):
        if use_cuda:
            index2 = torch.randperm(batch_size).cuda()
        else:
            index2 = torch.randperm(batch_size)
        mixed_x[i] =lam* x[i][index_dc] + (1 -lam) * x[i][index2][index_dc]
    return mixed_x, y[index_dc], lam


def mixup_data2(x, y, alpha=0.3, use_cuda=True):
    mixed_x={}
    batch_size = y.size()[0]
    for i in range(len(x) ):

        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
            #   剔除同类对
        mixed_x[i] =  x[i][index]

    return mixed_x, y[index]


def main(args, device):
    def train(model, train_loader,test_loader,  device, pre_epochs=20, epochs=50):
        model = model.to(device)

        optimizer = torch.optim.Adam([
            {'params': (p for n, p in model.named_parameters() if p.requires_grad and 'weight' in n),
             'weight_decay': 1e-2},
            {'params': (p for n, p in model.named_parameters() if p.requires_grad and 'weight' not in n)},
        ], lr=0.01)
        step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=23, gamma=0.1)
        valid_ccr=0.0
        best_valid_acc=0.0
        best_valid_ccr=0.0
        loss_epoch=[]
        criterion= torch.nn.CrossEntropyLoss()
        euclidean_dist_obj = torch.nn.PairwiseDistance(p=2).to(device)
        for epoch in range(1,args.num_epoch+1):
            model.train()
            train_loss, correct, num_samples = 0, 0, 0
            train_pred_label=[]
            train_Z=[]
            train_y=[]
            output=[]
            for batch in train_loader:
                x, y, index = batch['x'], batch['y'], batch['index']
                for k in x.keys():
                    x[k] = x[k].to(device)
                y = y.long().to(device)


                x_mix, y_mix, lam = mixup_data(x, y)

                x_all={}
                for i in range(len(x)):
                    x_all[i] = torch.cat([x[i], x_mix[i]], dim=0)

                evidence, TCPConfidence = model(x_all, y)


                prob = F.softmax(evidence, dim=1)
                p_target = torch.gather(input= prob, dim=1, index=y.unsqueeze(dim=1)).view(-1)[:y.shape[0]]


                confidence_loss=0
                for view in range(n_view):
                    confidence_loss += F.mse_loss(TCPConfidence[view].view(-1)[:y.shape[0]], p_target)
                loss_Oen=0
                for c in range(num_classes):
                    target_c= torch.LongTensor(y_mix.shape[0]).random_(c, c+ 1).to(device)
                    loss_Oen+= criterion(evidence[y.shape[0]:], target_c)



                loss = criterion(evidence[:y.shape[0]], y)



                loss += alpha*confidence_loss+ (1-lam)* beta*loss_Oen/num_classes + lam * beta*criterion(evidence[y.shape[0]:], y_mix)


                optimizer.zero_grad()


                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                train_y.append(torch.cat([y], dim=0).cpu().detach().numpy())
                train_Z.append(evidence[:y.shape[0]].cpu().detach().numpy())
                output.append(prob[:y.shape[0]].cpu().detach().numpy())

                num_samples += len(y)
                correct += torch.sum(prob.argmax(dim=-1)[:y.shape[0]].eq(y)).item()


            train_loss = train_loss / num_samples
            train_acc = correct / num_samples
            train_y = np.concatenate(train_y)
            train_Z = np.concatenate(train_Z)

            loss_epoch.append(train_loss)


            valid_Z,label,valid_ccr,valid_acc = valid(model, valid_loader, device)
            if valid_ccr:
                if best_valid_ccr < valid_ccr:
                    best_valid_ccr = valid_ccr
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_train_y = train_y
                    best_train_Z = train_Z
                    best_valid_Z = valid_Z
                    best_valid_label = [NCLASSES if i == unseen_label_index else i for i in label]

            else:
                valid_ccr = 0.0
            step_lr.step()
            print(
                f'Epoch {epoch:3d}: lr {step_lr.get_last_lr()[0]:.6f}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, valid ccr: {valid_ccr:.4f}')



        model.load_state_dict(best_model_wts)
        test_Z, softmax_scores, test_truth_label = test(model, test_loader, device)

        test_truth_label_open = [NCLASSES if i == unseen_label_index else i for i in test_truth_label]
        softmax_pred_label = np.argmax(softmax_scores, 1)
        softmax_ACC = metrics.accuracy_score(test_truth_label_open,
                                             softmax_pred_label)
        fpr_values, softmax_ccrs = EvaluationMetrics.ccr_at_fpr(
            np.array(test_truth_label_open),
            softmax_scores,
            NCLASSES)

        output_lines = ['CCR@FPR of {}% = {}'.format(fpr * 100, ccr) for fpr, ccr in
                        zip(fpr_values, softmax_ccrs)]
        output = ', '.join(output_lines)
        with open(args.file, "a") as f:
            f.write(
                'openness:{},unseen_num:{},dataset:{}\n'.format(args.openness, args.unseen_num, args.data))
            f.write(output)




        return model
    def valid(model, loader, device):
            model.eval()
            valid_Z=[]
            with torch.no_grad():
                label = []
                test_prob = []
                correct, num_samples = 0, 0
                for batch in loader:
                    x, y = batch['x'], batch['y']
                    for k in x.keys():
                        x[k] = x[k].to(device)

                    evidence = model.infer(x)
                    valid_Z.append(evidence.cpu().numpy())
                    prob = F.softmax(evidence, 1)
                    test_prob.append(prob.cpu().numpy())
                    pred_y=prob.argmax(dim=-1)
                    correct += torch.sum( pred_y.cpu().eq(batch['y'])).item()
                    num_samples += len(batch['y'])
                    label.append(batch['y'].cpu().numpy())
            prob = np.concatenate(test_prob)
            label = np.concatenate(label)
            valid_Z = np.concatenate(valid_Z)
            test_truth_label_open = [NCLASSES if i == unseen_label_index else i for i in label]
            fprs, softmax_ccrs = EvaluationMetrics.ccr_at_fpr(np.array(test_truth_label_open),
                                                                                  prob,
                                                                                  NCLASSES)

            return   valid_Z,label,softmax_ccrs[-1],correct/num_samples
    def test(model, loader, device):
            model.eval()
            with torch.no_grad():
                test_Z = []
                test_prob=[]
                label = []
                correct, num_samples = 0, 0
                for batch in loader:
                    x, y = batch['x'], batch['y']
                    for k in x.keys():
                        x[k] = x[k].to(device)

                    evidence = model.infer(x)
                    prob = F.softmax(evidence, 1)
                    test_Z.append(evidence.cpu().numpy())
                    test_prob.append(prob.cpu().numpy())
                    label.append(batch['y'].cpu().numpy())
                    pred_y=prob.argmax(dim=-1)
                    correct += torch.sum( pred_y.cpu().eq(batch['y'])).item()
                    num_samples += len(batch['y'])
            label = np.concatenate(label)
            prob = np.concatenate(test_prob)
            test_Z = np.concatenate(test_Z)
            return  test_Z,prob,label

    pdmf = SparseInferenceModule(n_feats, n_view,num_classes, device)
    print('---------------------------- Experiment ------------------------------')
    print('Number of views:', len(train_data.x), ' views with dims:', [v.shape[-1] for v in train_data.x.values()])
    print('Number of training samples:', len(train_data))
    print('Number of validating samples:', len(test_data))
    print('Trainable Parameters:')
    for n, p in pdmf.named_parameters():
        print('%-40s' % n, '\t', p.data.shape)
    print('----------------------------------------------------------------------')
    train(pdmf, train_loader,test_loader,device)

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--num_epoch', type=int, default=100, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--lambda-epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate')
    parser.add_argument("--save_file", type=str, default="res_TMC0312.txt")
    parser.add_argument('--save_results', default=True)
    parser.add_argument('--fix_seed', default=True)
    parser.add_argument('--T', default=10)
    args = parser.parse_args()



    args.device = '0'
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)


    dataset_dict = {1: 'esp_game', 2: 'flower17', 3: 'MNIST', 4: 'ORL',
                    5: 'smallReuters', 6: 'Youtube'}
    select_dataset = [1,2,3,4,5,6]


    training_rate = 0.1
    valid_rate = 0.1
    unseen_label_index = -100




    args.openness = 0.1

    path = './result/openness%.1f' % (args.openness)
    args.path = path


    for ii in select_dataset:
            data = dataset_dict[ii]
            args.data = data

            if data=="esp_game":
                alpha = 10
                beta = 1
            elif data=="Youtube":
                alpha = 0.5
                beta = 1

            elif data == "ORL" :
                alpha = 0.5
                beta = 0.1
            elif data == "MNIST":
                alpha = 1
                beta = 10
            elif data == "flower17":
                alpha = 0.5
                beta = 1
            elif data == "smallReuters":
                alpha = 1
                beta = 0.1
            args.file = f'./result/openness{args.openness:.1f}.txt'


            features, labels= load_data(dataset_dict[ii], './data/')
            n_view = len(features)
            n_feats = [x.shape[1] for x in features]
            n = features[0].shape[0]
            n_classes = len(np.unique(labels))

            print(data, n, n_view, n_feats)
            open2 = (1 - args.openness) * (1 - args.openness)
            args.unseen_num = round((1 - open2 / (2 - open2)) * n_classes)
            print("unseen_num:%d" % args.unseen_num)
            original_num_classes = len(np.unique(labels))
            seen_labels = list(range(original_num_classes - args.unseen_num))
            y_true = reassign_labels(labels, seen_labels, unseen_label_index)


            train_indices, test_valid_indices = special_train_test_split(y_true, unseen_label_index,
                                                                         test_size=1 - training_rate - valid_rate)
            valid_indices, test_indices = generate_partition(y_true[test_valid_indices], test_valid_indices,
                                                             valid_rate / (1 - training_rate))
            num_classes = np.max(y_true)+1
            NCLASSES=num_classes

            print(
                f'data:{data}\tseen_labels:{seen_labels}\tunseen_num:{args.unseen_num}\tnum_classes:{num_classes},split:{len(train_indices)}/{len(valid_indices)}/{len(test_indices)}')

            train_data = Multi_view_data( n_view,train_indices,features, y_true)
            valid_data = Multi_view_data( n_view, valid_indices, features, y_true)
            test_data = Multi_view_data( n_view, test_indices, features, y_true)


            train_loader = torch.utils.data.DataLoader(train_data
                                                       , batch_size=args.batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_data
                                                       , batch_size=args.batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_data
                                                       , batch_size=args.batch_size, shuffle=False)
            labels = torch.from_numpy(labels).long().to(device)
            y_true = torch.from_numpy(y_true).to(device)
            train_indices = torch.LongTensor(train_indices).to(device)

            N_mini_batches = len(train_loader)

            args.num_classes = num_classes
            args.seen_labels = seen_labels
            args.unseen_label_index = unseen_label_index

            if args.fix_seed:
                seed=20
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)


            print("========================", data)
            main(args, device)
            with open(args.file, "a") as f:
                    f.write('\n')
