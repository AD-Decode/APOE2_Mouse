import os
import numpy as np
import argparse
import time
import copy

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from imports.ABIDEDataset import ABIDEDataset
from torch_geometric.data import DataLoader
from net.braingnn_age_vertex import Network
from imports.utils import train_val_test_split
from sklearn.metrics import classification_report, confusion_matrix

import sys

import time

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def run_gnn(num_folds, current_fold):


    torch.manual_seed(123)

    EPS = 1e-10
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    #device = torch.device("cuda")

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=3, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='/home/alex/hsm/GNN/BrainGNN_Pytorch/data/APOE2/full_connect/age_vertex_14_sd', help='root directory of the dataset')
    parser.add_argument('--fold', type=int, default=0, help='training which fold')
    parser.add_argument('--lr', type = float, default=0.00005, help='learning rate')
    parser.add_argument('--stepsize', type=int, default=50, help='scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
    parser.add_argument('--weightdecay', type=float, default=5e-4, help='regularization')
    parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
    parser.add_argument('--lamb1', type=float, default=0, help='s1 unit regularization')
    parser.add_argument('--lamb2', type=float, default=0, help='s2 unit regularization')
    parser.add_argument('--lamb3', type=float, default=0.1, help='s1 entropy regularization')
    parser.add_argument('--lamb4', type=float, default=0.1, help='s2 entropy regularization')
    parser.add_argument('--lamb5', type=float, default=0.1, help='s1 consistence regularization')
    parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
    parser.add_argument('--ratio', type=float, default=0.3, help='pooling ratio')
    parser.add_argument('--indim', type=int, default=14, help='feature dim')
    parser.add_argument('--nroi', type=int, default=14, help='num of ROIs')
    parser.add_argument('--nclass', type=int, default=2, help='num of classes')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
    parser.add_argument('--save_path', type=str, default='./model/', help='path to save model')
    opt = parser.parse_args()

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    #################### Parameter Initialization #######################
    path = opt.dataroot
    name = 'APOE2'
    save_model = opt.save_model
    #load_model = opt.load_model
    opt_method = opt.optim
    num_epoch = opt.n_epochs
    fold = opt.fold
    writer = SummaryWriter(os.path.join('./log',str(fold)))

    #print(path)
    #print(name)

    ################## Define Dataloader ##################################

    dataset = ABIDEDataset(path,name)


    attrs = vars(dataset)
    print(attrs)
    #print(', '.join("%s: %s" % item for item in attrs.items()))

    #print(np.shape(dataset.data.y))
    print(np.shape(dataset.data.x))

    #print((dataset.data.y))
    dataset.data.y = dataset.data.y.squeeze()

    #dataset.data.y[dataset.data.y == float('inf')] = 0
    #dataset.data.y[dataset.data.y == float('nan')] = 0
    dataset.data.x[dataset.data.x == float('inf')] = 0

    #dataset.data.x[dataset.data.x == 0] = 0.0001

    print((dataset.data.x))

    np.save('age_x.npy',dataset.data.x)
    np.save('age_edge_index.npy',dataset.data.edge_index)
    np.save('age_edge_attr.npy',dataset.data.edge_attr)
    #np.save('age_batch.npy',dataset.data.batch)
    np.save('age_pos.npy',dataset.data.pos)


    history_train = np.zeros([4, num_epoch+1])

    ############### Define Graph Deep Learning Network ##########################
    model = Network(opt.indim,opt.ratio,opt.nclass).to(device)
    print(model)

    if opt_method == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
    elif opt_method == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr, momentum = 0.9, weight_decay=opt.weightdecay, nesterov = True)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

    ############################### Define Other Loss Functions ########################################
    def topk_loss(s,ratio):
        if ratio > 0.5:
            ratio = 1-ratio
        s = s.sort(dim=1).values
        res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
        return res


    def consist_loss(s):
        if len(s) == 0:
            return 0
        s = torch.sigmoid(s)
        W = torch.ones(s.shape[0],s.shape[0])
        D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
        L = D-W
        L = L.to(device)
        res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
        return res

    ###################### Network Training Function#####################################
    def train(epoch):
        print('train...........')
        scheduler.step()

        for param_group in optimizer.param_groups:
            print("LR", param_group['lr'])
        model.train()
        s1_list = []
        s2_list = []
        loss_all = 0
        step = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output, w1, w2, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
            s1_list.append(s1.view(-1).detach().cpu().numpy())
            s2_list.append(s2.view(-1).detach().cpu().numpy())
            loss_c = F.nll_loss(output, data.y)

            loss_p1 = (torch.norm(w1, p=2)-1) ** 2
            loss_p2 = (torch.norm(w2, p=2)-1) ** 2
            loss_tpk1 = topk_loss(s1,opt.ratio)
            loss_tpk2 = topk_loss(s2,opt.ratio)
            loss_consist = 0
            for c in range(opt.nclass):
                loss_consist += consist_loss(s1[data.y == c])
            loss = opt.lamb0*loss_c + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
                       + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5* loss_consist
            writer.add_scalar('train/classification_loss', loss_c, epoch*len(train_loader)+step)
            writer.add_scalar('train/unit_loss1', loss_p1, epoch*len(train_loader)+step)
            writer.add_scalar('train/unit_loss2', loss_p2, epoch*len(train_loader)+step)
            writer.add_scalar('train/TopK_loss1', loss_tpk1, epoch*len(train_loader)+step)
            writer.add_scalar('train/TopK_loss2', loss_tpk2, epoch*len(train_loader)+step)
            writer.add_scalar('train/GCL_loss', loss_consist, epoch*len(train_loader)+step)
            step = step + 1

            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()

            s1_arr = np.hstack(s1_list)
            s2_arr = np.hstack(s2_list)
        return loss_all / len(train_dataset), s1_arr, s2_arr ,w1,w2


    ###################### Network Testing Function#####################################
    def test_acc(loader):
        model.eval()
        correct = 0
        for data in loader:
            data = data.to(device)
            outputs= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
            pred = outputs[0].max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()

        return correct / len(loader.dataset)

    def test_loss(loader,epoch):
        print('testing...........')
        model.eval()
        loss_all = 0
        for data in loader:
            data = data.to(device)
            output, w1, w2, s1, s2= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
            loss_c = F.nll_loss(output, data.y)

            loss_p1 = (torch.norm(w1, p=2)-1) ** 2
            loss_p2 = (torch.norm(w2, p=2)-1) ** 2
            loss_tpk1 = topk_loss(s1,opt.ratio)
            loss_tpk2 = topk_loss(s2,opt.ratio)
            loss_consist = 0
            for c in range(opt.nclass):
                loss_consist += consist_loss(s1[data.y == c])
            loss = opt.lamb0*loss_c + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
                       + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5* loss_consist

            loss_all += loss.item() * data.num_graphs
        return loss_all / len(loader.dataset)

    #######################################################################################
    ############################   Model Training #########################################
    #######################################################################################




    tr_index_arr,_,te_index_arr = train_val_test_split(kfold=num_folds, fold=current_fold)

    tr_index = tr_index_arr.tolist()
    val_index = te_index_arr.tolist()
    te_index = te_index_arr.tolist()
    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[te_index]
    train_loader = DataLoader(train_dataset,batch_size=opt.batchSize, shuffle= True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)

    print(tr_index)
    print(te_index)

#        best_model_wts = copy.deepcopy(model.state_dict())
#        best_loss = 1e10
    for epoch in range(0, num_epoch):
        since  = time.time()
        tr_loss, s1_arr, s2_arr, w1, w2 = train(epoch)
        tr_acc = test_acc(train_loader)
        val_acc = test_acc(val_loader)
        val_loss = test_loss(val_loader,epoch)
        time_elapsed = time.time() - since
        print('*====**')
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Epoch: {:03d}, Train Loss: {:.7f}, '
              'Train Acc: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}'.format(epoch, tr_loss,
                                                           tr_acc, val_loss, val_acc))

        writer.add_scalars('Acc',{'train_acc':tr_acc,'val_acc':val_acc},  epoch)
        writer.add_scalars('Loss', {'train_loss': tr_loss, 'val_loss': val_loss},  epoch)
        history_train[0,num_epoch] = tr_loss
        history_train[1,num_epoch] = val_loss
        history_train[2,num_epoch] = tr_acc
        history_train[3,num_epoch] = val_acc


        #print(s1_arr)
        #print(s2_arr)
        #writer.add_histogram('Hist/hist_s1', s1_arr, epoch)
        #writer.add_histogram('Hist/hist_s2', s2_arr, epoch)

# Turning off best model weights


        if  epoch == num_epoch-1:
            print("saving model")
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            if save_model:
                torch.save(best_model_wts, os.path.join(opt.save_path,str(fold)+'.pth'))

    #######################################################################################
    ######################### Testing on testing set ######################################
    #######################################################################################

    if opt.load_model:
        model = Network(opt.indim,opt.ratio,opt.nclass).to(device)
        model.load_state_dict(torch.load(os.path.join(opt.save_path,str(fold)+'.pth')))
        model.eval()
        preds = []
        correct = 0
        for data in val_loader:
            data = data.to(device)
            outputs= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
            pred = outputs[0].max(1)[1]
            preds.append(pred.cpu().detach().numpy())
            correct += pred.eq(data.y).sum().item()
        preds = np.concatenate(preds,axis=0)
        trues = val_dataset.data.y.cpu().detach().numpy()
        cm = confusion_matrix(trues,preds)
        print("Confusion matrix")
        print(classification_report(trues, preds))

    else:
       model.load_state_dict(best_model_wts)
       model.eval()
       test_accuracy = test_acc(test_loader)
       test_l= test_loss(test_loader,0)
       print("===========================")
       print("Test Acc: {:.7f}, Test Loss: {:.7f} ".format(test_accuracy, test_l))
       print(opt)
    #kfold_results[0] = len(te_index)
    #kfold_results[1] = test_accuracy
    #print(kfold_results[i,:])

#total_acc = np.sum(kfold_results[0,:]*kfold_results[1,:])/33
#print('Total Test Acc of 10 fold: ' + str(total_acc))
    np.save('train_loss_'+ str(current_fold) +'.npy', history_train)
    return len(te_index), tr_acc, test_accuracy



if __name__=='__main__':
    start_timer = time.perf_counter()
    num_folds = 5
    results = np.zeros([3,num_folds])
    results[0,0],results[1,0],results[2,0] = run_gnn(num_folds,0)
    results[0,1],results[1,1],results[2,1] = run_gnn(num_folds,1)
    results[0,2],results[1,2],results[2,2] = run_gnn(num_folds,2)
    results[0,3],results[1,3],results[2,3] = run_gnn(num_folds,3)
    results[0,4],results[1,4],results[2,4] = run_gnn(num_folds,4)
    #results[0,5], results[1,5] = run_gnn(num_folds,5)
    #results[0,6], results[1,6] = run_gnn(num_folds,6)
    #results[0,7], results[1,7] = run_gnn(num_folds,7)
    #results[0,8], results[1,8] = run_gnn(num_folds,8)
    #results[0,9], results[1,9] = run_gnn(num_folds,9)
    print(np.sum(results[0,:]*results[1,:])/33)
    print(np.sum(results[0,:]*results[2,:])/33)
    np.save('test_acc_results_age_14_sd_5fold_test.npy', results)
    end_timer = time.perf_counter()
    dur = end_timer - start_timer
    print('Time elapsed: ' + str(dur/60) + ' min')









