# -*- coding:utf-8 -*-
# 防止注释中出现中文运行报错

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import time
import torch
import csv
import os
import random
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, cohen_kappa_score
import torch.utils.data as Data
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import logging
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.autograd import Variable  
# from rdkit import Chem
# from rdkit.Chem import rdchem
# import dgl
# import torch as th
# from torch.nn import init
# from molecular_graph_dgl_egat import EGATModule, smiles_to_dgl_graph
import warnings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings("ignore")



def get_drugpair_info(dir, list, drugs):
    with open(dir) as raw_input_file:
        data = csv.reader(raw_input_file, delimiter=',')
        header = next(data)
        if multi_class == True:
            for p, r in data:
                list.append([eval(p), eval(
                    r)])
                if eval(p)[
                    0] not in drugs:
                    drugs.append(eval(p)[0])
                if eval(p)[1] not in drugs:
                    drugs.append(eval(p)[1])
            return list, drugs
        else:
            for p, r in data:
                x = [eval(p), *eval(r)]
                list.append(x)
                if eval(p)[0] not in drugs:
                    drugs.append(eval(p)[0])
                if eval(p)[1] not in drugs:
                    drugs.append(eval(p)[1])
            return list, drugs



def feature_vector(feature_dir, drugs):
    feature = {}
    with open(feature_dir) as file1:
        data = csv.reader(file1)

        header = next(data)
        if feature_dir != filename[4]:
            for d, emb in data:
                if d in drugs:

                    feature[d] = torch.tensor(eval(emb), dtype=torch.float32)
        else:
            for d, smiles in data:
                if d in drugs:
                    feature[d] = smiles
    return feature



def train_test_data1(data_lis):
    train_X_data=[]
    train_Y_data=[]
    test_X_data=[]
    test_Y_data=[]
    data_lis=np.array(data_lis,dtype=object)
    if multi_class == True:
        drug_pair=data_lis[:,0]
        Y=data_lis[:,1]
        label=np.array(list(map(int,Y)))
    else:
        drug_pair=data_lis[:,0]
        Y=data_lis[:,1:201]
        label=Y

    if multi_class:
        kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=3)
    else:
        kfold=KFold(n_splits=5,shuffle=True,random_state=3)
    for train,test in kfold.split(drug_pair,label):
        train_X_data.append(drug_pair[train])
        train_Y_data.append(label[train])
        test_X_data.append(drug_pair[test])
        test_Y_data.append(label[test])
    train_X=np.array(train_X_data,dtype=object)
    train_Y=np.array(train_Y_data,dtype=object)
    test_X=np.array(test_X_data,dtype=object)
    test_Y=np.array(test_Y_data,dtype=object)
    return train_X,train_Y,test_X,test_Y


def create_log_id(dir_path):
    log_count = 0
    file_path = os.path.join(dir_path, 'log{:d}'.format(
        log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, 'log{:d}'.format(
            log_count))
    return log_count



def logging_config(folder=None, name=None, level=logging.DEBUG, console_level=logging.DEBUG, no_console=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".txt")
    print("All logs will be saved to %s" % logpath)

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder


def early_stopping(recall_list, stopping_steps, min_epoch):
    best_recall = max(recall_list)
    best_step = recall_list.index(best_recall)
    if len(recall_list) - best_step - 1 >= stopping_steps and min_epoch > 70:
        should_stop = True
    else:
        should_stop = False
    return best_recall, should_stop




def extract_features(batch_data, kg_features, fg_features, seq_features):

    kg1_list, fg1_list, seq1_list = [], [], []
    kg2_list, fg2_list, seq2_list = [], [], []


    for i in batch_data:
        drug1, drug2 = i[0], i[1]



        kg1_list.append(kg_features[drug1].to(device))
        fg1_list.append(fg_features[drug1].to(device))
        seq1_list.append(seq_features[drug1].to(device))


        kg2_list.append(kg_features[drug2].to(device))
        fg2_list.append(fg_features[drug2].to(device))
        seq2_list.append(seq_features[drug2].to(device))


    kg1 = torch.stack(kg1_list)
    fg1 = torch.stack(fg1_list)
    seq1 = torch.stack(seq1_list)
    kg2 = torch.stack(kg2_list)
    fg2 = torch.stack(fg2_list)
    seq2 = torch.stack(seq2_list)

    return kg1, fg1, seq1, kg2, fg2, seq2




class CrossAttention_wproj(nn.Module):
    def __init__(self, hidden_dim, num_heads):

        super().__init__()


        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.norm = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(1024, 3072),
            nn.ReLU(),
            nn.Linear(3072, 1024),
        )


        self.weight_mlp = nn.Sequential(
            nn.Linear(1024, 3072),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(3072, 1)
        )

    def forward(self, q_feats, k_feats, v_feats):

        q = q_feats.unsqueeze(1)
        k = k_feats.unsqueeze(1)
        v = v_feats.unsqueeze(1)


        attn_output, _ = self.multihead_attn(q, k, v)
        attn_output = attn_output.squeeze(1)


        I = torch.stack([k_feats, attn_output], dim=1)


        scores = self.weight_mlp(I)
        weights = torch.softmax(scores, dim=1)


        weighted_sum = (weights * I).sum(dim=1)


        output_AN = self.norm(weighted_sum)
        output_FA = output_AN + self.ffn(output_AN)

        return output_FA

class MyModel(nn.Module):
    def __init__(self, kg_features, fg_features, seq_features):
        super(MyModel, self).__init__()

        self.kg_features = kg_features
        self.fg_features = fg_features
        self.seq_features = seq_features

        if multi_class == True:
            self.fea_dim = 67
        else:
            self.fea_dim = 200



        self.cross_attn2 = CrossAttention_wproj(
            hidden_dim=1024,
            num_heads=4)


        self.fc1 = nn.Sequential(nn.Linear(3072, 4096), nn.BatchNorm1d(4096), nn.Dropout(0.2), nn.ReLU(True))

        self.fc2 = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.Dropout(0.2), nn.ReLU(True))
        self.norm1 = nn.LayerNorm(1024)
        self.norm_intra_1_q = nn.LayerNorm(400)
        self.norm_intra_1_k = nn.LayerNorm(400)
        self.norm_intra_1_v = nn.LayerNorm(800)

        self.q_proj = nn.Linear(400, 1024)
        # self.k_proj = nn.Linear(400, 1024)
        self.v_proj = nn.Linear(800, 1024)

        self.weight_mlp = nn.Sequential(
            nn.Linear(1024, 3072),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(3072, 1)
        )


        self.classifier_fc1 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.1),
            nn.ReLU(True)
        )

        self.classifier_fc2 = nn.Sequential(
            nn.Linear(2048, 3072),
            nn.BatchNorm1d(3072),
            nn.Dropout(0.1),
            nn.ReLU(True)
        )

        self.classifier_fc3 = nn.Sequential(
            nn.Linear(3072, 4096),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.1),
            nn.ReLU(True)
        )

        self.classifier_fc4 = nn.Sequential(nn.Linear(4096, self.fea_dim))




    def train_DDI_data(self, mode, train_data):
        kg1, fg1, seq1, kg2, fg2, seq2 = extract_features(train_data, self.kg_features, self.fg_features, self.seq_features)

        dg1_intra_1 = self.cross_attn2(self.q_proj(self.norm_intra_1_q(fg1)), self.q_proj(self.norm_intra_1_k(fg1)), self.v_proj(self.norm_intra_1_v(torch.cat((seq1, kg1), dim=1))))
        dg2_intra_1 = self.cross_attn2(self.q_proj(self.norm_intra_1_q(fg2)), self.q_proj(self.norm_intra_1_k(fg2)), self.v_proj(self.norm_intra_1_v(torch.cat((seq2, kg2), dim=1))))


        dg1_inter_1 = self.cross_attn2(self.norm1(dg1_intra_1), self.norm1(dg2_intra_1), self.norm1(dg2_intra_1))
        dg2_inter_1 = self.cross_attn2(self.norm1(dg2_intra_1), self.norm1(dg1_intra_1), self.norm1(dg1_intra_1))




        dg1_intra_2 = self.cross_attn2(self.norm1(dg1_inter_1), self.norm1(dg1_intra_1), self.norm1(dg1_intra_1))
        dg2_intra_2 = self.cross_attn2(self.norm1(dg2_inter_1), self.norm1(dg2_intra_1), self.norm1(dg2_intra_1))


        dg1_inter_2 = self.cross_attn2(self.norm1(dg1_intra_2), self.norm1(dg2_intra_2), self.norm1(dg2_intra_2))
        dg2_inter_2 = self.cross_attn2(self.norm1(dg2_intra_2), self.norm1(dg1_intra_2), self.norm1(dg1_intra_2))


        dg1 = torch.stack([self.norm1(dg1_inter_1), self.norm1(dg1_inter_2)], dim=1)
        scores_1 = self.weight_mlp(dg1)
        weights_1 = torch.softmax(scores_1, dim=1)
        dg1_sum = (weights_1 * dg1).sum(dim=1)

        dg2 = torch.stack([self.norm1(dg2_inter_1), self.norm1(dg2_inter_2)], dim=1)
        scores_2 = self.weight_mlp(dg2)
        weights_2 = torch.softmax(scores_2, dim=1)
        dg2_sum = (weights_2 * dg2).sum(dim=1)


        dg1_final = self.fc2(self.fc1(torch.cat((dg1_sum, self.norm1(dg1_inter_1), self.norm1(dg1_inter_2)), dim=1)))
        dg2_final = self.fc2(self.fc1(torch.cat((dg2_sum, self.norm1(dg2_inter_1), self.norm1(dg2_inter_2)), dim=1)))

        drugpair = torch.cat((dg1_final, dg2_final), dim=1)

        out = self.classifier_fc4(self.classifier_fc3(self.classifier_fc2(self.classifier_fc1(drugpair))))

        return out

    def test_DDI_data(self, mode, test_data):
        kg1, fg1, seq1, kg2, fg2, seq2 = extract_features(test_data, self.kg_features, self.fg_features, self.seq_features)

        dg1_intra_1 = self.cross_attn2(self.q_proj(self.norm_intra_1_q(fg1)), self.q_proj(self.norm_intra_1_k(fg1)), self.v_proj(self.norm_intra_1_v(torch.cat((seq1, kg1), dim=1))))
        dg2_intra_1 = self.cross_attn2(self.q_proj(self.norm_intra_1_q(fg2)), self.q_proj(self.norm_intra_1_k(fg2)), self.v_proj(self.norm_intra_1_v(torch.cat((seq2, kg2), dim=1))))


        dg1_inter_1 = self.cross_attn2(self.norm1(dg1_intra_1), self.norm1(dg2_intra_1), self.norm1(dg2_intra_1))
        dg2_inter_1 = self.cross_attn2(self.norm1(dg2_intra_1), self.norm1(dg1_intra_1), self.norm1(dg1_intra_1))




        dg1_intra_2 = self.cross_attn2(self.norm1(dg1_inter_1), self.norm1(dg1_intra_1), self.norm1(dg1_intra_1))
        dg2_intra_2 = self.cross_attn2(self.norm1(dg2_inter_1), self.norm1(dg2_intra_1), self.norm1(dg2_intra_1))


        dg1_inter_2 = self.cross_attn2(self.norm1(dg1_intra_2), self.norm1(dg2_intra_2), self.norm1(dg2_intra_2))
        dg2_inter_2 = self.cross_attn2(self.norm1(dg2_intra_2), self.norm1(dg1_intra_2), self.norm1(dg1_intra_2))


        dg1 = torch.stack([self.norm1(dg1_inter_1), self.norm1(dg1_inter_2)], dim=1)
        scores_1 = self.weight_mlp(dg1)
        weights_1 = torch.softmax(scores_1, dim=1)
        dg1_sum = (weights_1 * dg1).sum(dim=1)

        dg2 = torch.stack([self.norm1(dg2_inter_1), self.norm1(dg2_inter_2)], dim=1)
        scores_2 = self.weight_mlp(dg2)
        weights_2 = torch.softmax(scores_2, dim=1)
        dg2_sum = (weights_2 * dg2).sum(dim=1)

        dg1_final = self.fc2(self.fc1(torch.cat((dg1_sum, self.norm1(dg1_inter_1), self.norm1(dg1_inter_2)), dim=1)))
        dg2_final = self.fc2(self.fc1(torch.cat((dg2_sum, self.norm1(dg2_inter_1), self.norm1(dg2_inter_2)), dim=1)))

        drugpair = torch.cat((dg1_final, dg2_final), dim=1)

        out = self.classifier_fc4(self.classifier_fc3(self.classifier_fc2(self.classifier_fc1(drugpair))))
        if multi_class == True:
            sm = nn.Softmax(dim=1)
            pre = sm(out)
        else:
            sg = nn.Sigmoid()
            pre = sg(out)
        return pre

    def forward(self, mode, *input):
        if mode == 'train':
            return self.train_DDI_data(mode, *input)
        if mode == 'test':
            return self.test_DDI_data(mode, *input)


def calc_metrics(y_true, y_pred, pred_score):
    if multi_class == True:
        acc = accuracy_score(y_true, y_pred)
        macro_precision = precision_score(y_true, y_pred, average='macro')
        # macro_recall = recall_score(y_true, y_pred, average='macro')
        # macro_f1 = f1_score(y_true, y_pred, average='macro')
        kappa = cohen_kappa_score(y_true, y_pred)
        y_true_bi = F.one_hot(y_true.to(torch.int64), num_classes=67)

        auc_ = roc_auc_score(y_true_bi, pred_score)
        aupr = average_precision_score(y_true_bi, pred_score)
        return acc, macro_precision, kappa, auc_, aupr
    else:
        auc_ = roc_auc_score(y_true, pred_score)
        aupr = average_precision_score(y_true, pred_score)
        return auc_, aupr



def pred_tru(loader_test, model):
    with torch.no_grad():
        for i, data in enumerate(loader_test):
            test_x_map = data[0]
            test_x = []
            for k in range(len(test_x_map[0])):
                dp = (test_x_map[0][k], test_x_map[1][k])
                test_x.append(dp)
            if multi_class == True:
                if i == 0:
                    test_y = data[1]
                else:
                    test_y = torch.cat((test_y, data[1]), 0)
            else:
                test_y1 = data[1].unsqueeze(
                    0)
                for k in range(2, 201):
                    test_y1 = torch.cat((test_y1, data[k].unsqueeze(0)), dim=0)
                test_y1 = test_y1.permute(1, 0)
                if i == 0:
                    test_y = test_y1
                else:
                    test_y = torch.cat((test_y, test_y1), 0)
            out1 = model('test', test_x)
            if i == 0:
                out = out1
            else:
                out = torch.cat((out, out1), 0)
    return out, test_y



def evaluate(loader_test, model):
    model.eval()

    with torch.no_grad():
        out, test_y = pred_tru(loader_test, model)

        prediction = torch.max(out, 1)[1]
        prediction = prediction.cuda().data.cpu().numpy()
        out = out.cuda().data.cpu().numpy()
        if multi_class == True:
            acc, macro_precision, kappa, auc_, aupr = calc_metrics(test_y, prediction, out)
            return macro_precision, acc, kappa, auc_, aupr
        else:
            auc_, aupr = calc_metrics(test_y.numpy(), prediction, out)
            return auc_, aupr



def pos_weight():
    data1 = []
    with open(filename[0]) as f2:
        data2 = csv.reader(f2)
        header = next(data2)
        for i, j in data2:
            data1.append(eval(j))
    data3 = torch.Tensor(data1)
    num = data3.size(0)
    posn = torch.sum(data3, 0)
    numn = torch.full_like(posn, num)
    pos_weight = torch.div(numn - posn, posn).to('cuda')
    return pos_weight


def Train(batch_size, n_epoch, kg_features, fg_features, seq_features):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    save_dir = 'log\\'
    if multi_class:
        logging_config(folder=save_dir, name='mc_log', no_console=False)
    else:
        logging_config(folder=save_dir, name='ml_log', no_console=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(42)

    macro_precision_list = []
    macro_recall_list = []
    macro_f1_list = []
    acc_list = []
    kappa_list = []
    auc_list = []
    aupr_list = []


    for i in range(5):
        print('第', i + 1, '折：')
        time0 = time.time()
        # '''
        model = MyModel(kg_features, fg_features, seq_features)
        model.to(device)
        logging.info(model)


        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
        scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.1)


        trainset = []
        # trainx=trainX[i].tolist()
        # trainy=trainY[i].tolist()
        trainx = trainX[i]
        trainy = trainY[i]
        for j in range(len(trainx)):
            if multi_class == True:
                trainset.append([trainx[j], trainy[j]])
            else:
                x1 = [trainx[j]]
                for k in range(200):
                    x1.append(trainy[j][k])
                trainset.append(x1)


        print(f'Trainset size: {len(trainset)}')


        loader_train = Data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

        print(f'Loader train length: {len(loader_train)}')


        print(f'trainX size: {len(trainX[i])}')
        print(f'trainY size: {len(trainY[i])}')


        testset = []
        # testx = testX[i].tolist()
        # testy = testY[i].tolist()
        testx = testX[i]
        testy = testY[i]
        for j in range(len(testx)):
            if multi_class == True:
                testset.append([testx[j], testy[j]])
            else:
                x11 = [testx[j]]
                for k in range(200):
                    x11.append(testy[j][k])
                testset.append(x11)
        loader_test = Data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

        best_epoch = -1
        val_list = []
        for epoch in range(1, n_epoch + 1):
            model.train()

            ddi_total_loss = 0
            for step, tdata in enumerate(loader_train):
                iter = step + 1
                time2 = time.time()
                train_x_map = tdata[0]
                if multi_class == True:
                    train_y = tdata[1]
                else:
                    train_y = tdata[1].unsqueeze(0)
                    for k in range(2, 201):
                        train_y = torch.cat((train_y, tdata[k].unsqueeze(0)), dim=0)
                    train_y = train_y.permute(1, 0)

                if use_cuda:
                    train_y = train_y.to(device)
                train_x = []
                for ii in range(len(train_x_map[0])):
                    dp = (train_x_map[0][ii], train_x_map[1][ii])
                    train_x.append(dp)
                out = model('train', train_x)

                if multi_class == True:
                    loss_func = torch.nn.CrossEntropyLoss()
                    loss = loss_func(out, train_y.long())
                else:
                    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    loss = loss_func(out, train_y.to(torch.float))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                ddi_total_loss += loss.item()
                if (iter % 100) == 0:
                    logging.info(
                        'DDI Training: Epoch {:04d} Iter {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'
                        .format(epoch, iter, time.time() - time2, loss.item(), ddi_total_loss / iter))


            scheduler.step()
            if multi_class == True:
                time3 = time.time()

                macro_precision, acc, kappa, auc_, aupr = evaluate(loader_test, model)
                logging.info(
                    'DDI Evaluation:Total Time {:.1f}s | Macro Precision {:.4f} | ACC {:.4f} | Kappa {:.4f} | AUC {:.4f} | AUPR {:.4f}'
                    .format(time.time() - time3, macro_precision, acc, kappa, auc_, aupr))
                val_list.append(macro_precision + acc + kappa + auc_ + aupr)
                best_acc, should_stop = early_stopping(val_list, 20, epoch)

                if should_stop:
                    # PATH = os.path.join('multi_class\\{}\\'.format(i+1), 'best_model_epoch.pth')
                    PATH = 'best_model_epoch.pth'
                    model.load_state_dict(torch.load(PATH))
                    model.to(device)
                    macro_precision, acc, kappa, auc_, aupr = evaluate(loader_test, model)
                    logging.info(
                        'Final DDI Evaluation:Macro Precision {:.4f} | ACC {:.4f} | Kappa {:.4f} | AUC {:.4f} | AUPR {:.4f}'
                        .format(macro_precision, acc, kappa, auc_, aupr))
                    break
                if val_list.index(best_acc) == len(val_list) - 1:
                    # PATH = os.path.join('multi_class\\{}\\'.format(i+1),'best_model_epoch.pth')
                    PATH = 'best_model_epoch.pth'
                    torch.save(model.state_dict(), PATH)
                    best_epoch = epoch

            else:
                time4 = time.time()

                auc_, aupr = evaluate(loader_test, model)
                logging.info('DDI Evaluation:Total Time {:.1f}s | AUC {:.4f} | AUPR {:.4f}'
                             .format(time.time() - time4, auc_, aupr))
                val_list.append(auc_ + aupr)

                best_auc, should_stop = early_stopping(val_list, 20, epoch)
                if should_stop:
                    # PATH = os.path.join('multi_label\\{}\\'.format(i+1),'best_model_epoch.pth')
                    PATH = 'best_model_epoch.pth'
                    model.load_state_dict(torch.load(PATH))
                    model.to(device)
                    auc_, aupr = evaluate(loader_test, model)
                    logging.info('Final DDI Evaluation:AUC {:.4f} | AUPR {:.4f}'
                                 .format(auc_, aupr))
                    break
                if val_list.index(best_auc) == len(val_list) - 1:
                    # PATH = os.path.join('multi_label\\{}\\'.format(i+1),'best_model_epoch.pth')
                    PATH = 'best_model_epoch.pth'
                    torch.save(model.state_dict(), PATH)
                    best_epoch = epoch

        print('第', i + 1, '折，best epoch:', best_epoch)
        # '''
        # test:
        if multi_class == True:
            # PATH = os.path.join('multi_class\\{}\\'.format(i+1), 'best_model_epoch.pth')
            PATH = 'best_model_epoch.pth'
        else:
            # PATH = os.path.join('multi_label\\{}\\'.format(i+1),'best_model_epoch.pth')
            PATH = 'best_model_epoch.pth'
        model.load_state_dict(torch.load(PATH))
        model.to(device)

        if multi_class == True:
            time5 = time.time()
            macro_precision, acc, kappa, auc_, aupr = evaluate(loader_test, model)
            logging.info(
                'DDI Test:Total Time {:.1f}s | Macro Precision {:.4f} | ACC {:.4f} | Kappa {:.4f} | AUC {:.4f} | AUPR {:.4f}'
                .format(time.time() - time5, macro_precision, acc, kappa, auc_, aupr))
            macro_precision_list.append(macro_precision)
            acc_list.append(acc)
            kappa_list.append(kappa)
            auc_list.append(auc_)
            aupr_list.append(aupr)

        else:
            time6 = time.time()
            auc_, aupr = evaluate(loader_test, model)
            logging.info('DDI Test:Total Time {:.1f}s | AUC {:.4f} | AUPR {:.4f}'
                         .format(time.time() - time6, auc_, aupr))
            auc_list.append(auc_)
            aupr_list.append(aupr)

        logging.info('Training+Evaluation:Total Time {:.1f}s '.format(time.time() - time0))

    if multi_class == True:
        mean_macro_precision = np.mean(macro_precision_list)
        mean_acc = np.mean(acc_list)
        mean_kappa = np.mean(kappa_list)
        mean_auc = np.mean(auc_list)
        mean_aupr = np.mean(aupr_list)
        logging.info(
            '5-fold cross validation DDI Mean Evaluation: Macro Precision {:.4f} | ACC {:.4f} | Kappa {:.4f} | AUC {:.4f} | AUPR {:.4f}'
            .format(mean_macro_precision, mean_acc, mean_kappa, mean_auc, mean_aupr))

    else:
        mean_auc = np.mean(auc_list)
        mean_aupr = np.mean(aupr_list)
        logging.info(
            '5-fold cross validation DDI Mean Evaluation: AUC {:.4f} | AUPR {:.4f} '.format(mean_auc, mean_aupr))


if __name__ == '__main__':

    multi_class = False

    if multi_class == True:
        filename = ['Multi_class_dataset.csv',
                    'drug_KG_features_max_hops_3.csv', 'drug_graph_features.csv', 'drug_smiles_simple_embedding_400D.csv','drug_smiles.csv']
    else:
        filename = ['Multi_label_dataset.csv',
                    'drug_KG_features_max_hops_3.csv', 'drug_graph_features.csv', 'drug_smiles_simple_embedding_400D.csv','drug_smiles.csv']
        pos_weight = pos_weight()

    data_list, drugs = get_drugpair_info(filename[0], [], [])
    kg_features = feature_vector(filename[1], drugs)
    fg_features = feature_vector(filename[2], drugs)
    seq_features = feature_vector(filename[3], drugs)
    smiles_list = feature_vector(filename[4], drugs)

    trainX, trainY, testX, testY = train_test_data1(data_list)
    print('数据划分over')

    Train(batch_size=1024, n_epoch=200, kg_features=kg_features, fg_features=fg_features, seq_features=seq_features)