import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import os.path
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim import Adam, lr_scheduler
from ranger import Ranger
from torch.utils import data
from train_test import train_one_epoch,test_one_epoch
from model_1 import Emotion_Net
from model_2 import VGG19_1D
from SENet_model import SENet_Emotion_Net
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold
import datetime
from EarlyStopping import EarlyStopping

import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
import random

import pandas as pd
#from callbacks import LossHistory
import warnings

warnings.filterwarnings('ignore')


# global random
def seed_torch(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # forbidden hash randomization
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('The random seed：', seed)
seed_torch()

begin_time = datetime.datetime.now()  # timing
fp_16 = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
enable_amp = True if "cuda" in device.type else False
if fp_16:
    print("Mixing automatically：", enable_amp)
model_path = ""
# training epoch setting
init_epoch = 0
epochs = 100
'''
WESAD
'''

eda_data = np.load(r"/mnt/bysj_2024/WESAD_three/EDA_3_256.npy", allow_pickle=True)
resp_data = np.load(r"/mnt/bysj_2024/WESAD_three/Resp_3_256.npy", allow_pickle=True)
emg_data = np.load(r"/mnt/bysj_2024/WESAD_three/EMG_3_256.npy", allow_pickle=True)
ecg_data = np.load(r"/mnt/bysj_2024/WESAD_three/ECG_3_256.npy", allow_pickle=True)
label = np.load(r"/mnt/bysj_2024/WESAD_three/label_3.npy", allow_pickle=True)


# normalization
eda_data = preprocessing.scale(eda_data)
resp_data = preprocessing.scale(resp_data)
emg_data = preprocessing.scale(emg_data)
ecg_data = preprocessing.scale(ecg_data)

# integration
phy_data = np.concatenate([ecg_data, emg_data,eda_data,resp_data], axis=-1)
label=label-1
print("Data shape：", phy_data.shape)
print("Label shape：", label.shape)
print(pd.DataFrame(label).describe())
print(pd.DataFrame(label).value_counts())

##divided to 90% 10%
x_train, x_test, y_train, y_test = train_test_split(phy_data, label, random_state=42, test_size=0.2)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=42, test_size=0.5)

i = 0
print("label shape:", label.shape)
print("x_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print("x_test shape", x_test.shape)
print("y_test shape", y_test.shape)
print("x_val shape", x_val.shape)
print("y_val shape", y_val.shape)

'''
# 10-fold
test_accuracy_allfold = np.zeros(shape=[0], dtype=float)
test_f1score_allfold = np.zeros(shape=[0], dtype=float)
train_used_time_allfold = np.zeros(shape=[0], dtype=float)
test_used_time_allfold = np.zeros(shape=[0], dtype=float)
kf = KFold(n_splits=10, shuffle=True, random_state=0)
n = 0
acc = []

save_testf1 = np.zeros(shape=[0], dtype=float)
save_testacc = np.zeros(shape=[0], dtype=float)
save_testloss = np.zeros(shape=[0], dtype=float)

# test
phydata_test = torch.from_numpy(x_test).type(torch.FloatTensor)
label_test = torch.from_numpy(y_test).type(torch.LongTensor)

test_dataset_final = data.TensorDataset(phydata_test, label_test)
test_loader_final = data.DataLoader(dataset=test_dataset_final, batch_size=128, shuffle=True, drop_last=True)
test_num = x_test.shape[0]

for train_index, val_index in kf.split(x_train):
    # import models
    model = Emotion_Net(num_class=3)
    #model = VGG19_1D(num_class=3)
    #model = SENet_Emotion_Net(num_class=3)
    model.to(device)
    n += 1
    # ------#
    # dairy
    train_log_filename = "train_log" + str(n) + "_fold.txt"
    result_dir = '/mnt/bysj_2024/logs/'
    train_log_filepath = os.path.join(result_dir, train_log_filename)
    train_log_txt_formatter = "{time_str} [Epoch] {epoch:04d} [Train_Loss] {train_loss_str} [Train_acc] {train_acc_str} [Test_Loss] {test_loss_str} [Test_acc] {test_acc_str} [max_acc] {max_acc_str} [f1] {f1_str} [max_f1] {max_f1}\n"
    train_accL, train_lossL, test_accL, test_lossL, f1L = [], [], [], [], []
    x_train_k = torch.from_numpy(x_train[train_index]).type(torch.FloatTensor)
    y_train_k = torch.from_numpy(y_train[train_index]).type(torch.LongTensor)
    x_val_k = torch.from_numpy(x_train[val_index]).type(torch.FloatTensor)
    y_val_k = torch.from_numpy(y_train[val_index]).type(torch.LongTensor)
    train_num, val_num = x_train_k.shape[0], x_val_k.shape[0]
    train_dataset_k = data.TensorDataset(x_train_k, y_train_k)
    val_dataset_k = data.TensorDataset(x_val_k, y_val_k)
    train_loader_k = data.DataLoader(dataset=train_dataset_k, batch_size=128, shuffle=True, drop_last=True)
    val_loader_k = data.DataLoader(dataset=val_dataset_k, batch_size=128, shuffle=True, drop_last=True)
    print("len(train_loader_k)", len(train_dataset_k))
    print("len(val_dataset_k)", len(val_dataset_k))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("The computation unit：", torch.cuda.get_device_name(0))
    criterion1 = nn.CrossEntropyLoss()
    criterion = [criterion1]
    #optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5, eps=1e-8)
    optimizer = Ranger(model.parameters())
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    # train
    train_start_time = time.time()
    train_used_time_fold = time.time() - train_start_time
    print('The' + str(n) + 'epochs')
    max_testacc = 0
    max_f1 = 0
    best_test_loss = 100
    model_savepath1 = '/mnt/bysj_2024/logs/model_save/'
    early_stopping = EarlyStopping(model_savepath1)
    # --------------------#
    #    record loss
    # ---------------------#
    save_dir = '/mnt/bysj_2024/logs/loss_history'
    loss_history = LossHistory(save_dir, model)
    for epoch in range(init_epoch, epochs):
        train_accL, train_lossL, test_accL, test_lossL, f1L, min_loss, es, max_testacc, max_f1, best_test_loss = train_one_epoch(
            model, criterion, optimizer, train_loader_k, train_num, epoch, epochs, val_loader_k, val_num,
            train_accL, train_lossL, test_accL, test_lossL, f1L, device, max_testacc, max_f1, best_test_loss,
            early_stopping, loss_history)
        # dairy
        to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                                  epoch=epoch + 1,
                                                  train_loss_str=" ".join(["{}".format(train_lossL[-1])]),  # train loss
                                                  train_acc_str=" ".join(["{}".format(train_accL[-1])]),  # train accuracy
                                                  test_loss_str=" ".join(["{}".format(test_lossL[-1])]),  # test loss
                                                  test_acc_str=" ".join(["{}".format(test_accL[-1])]),
                                                  max_acc_str=" ".join(["{}".format(max_testacc)]),
                                                  f1_str=" ".join(["{}".format(f1L[-1])]),
                                                  max_f1=" ".join(["{}".format(max_f1)]))  # test accuracy
        with open(train_log_filepath, "a") as f:
            f.write(to_write)
        if es:
            print("EarlyStop")
            break
    test_model = model
    test_model.to('cuda')
    model_dict = test_model.state_dict()
    pretrained_dict = torch.load('/mnt/bysj_2024/logs/model_save/best_model.pth', map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    test_model.load_state_dict(model_dict)
    test_acc, test_loss, f1 = test_one_epoch(test_model, criterion, test_loader_final,
                                                                          test_num, final_test=True, k=n)
    print('The' + str(n) + 'epochs  test_loss:', test_loss, 'Test_Acc：', test_acc, 'Test_f1score: ', f1)
    print("Time：", (datetime.datetime.now() - begin_time))
    save_testf1 = np.append(save_testf1, f1)
    save_testacc = np.append(save_testacc, test_acc)
    save_testloss = np.append(save_testloss, test_loss)

    np.save('/mnt/bysj_2024/logs/' + str(n) + 'fold_trainaccL.npy', train_accL)
    np.save('/mnt/bysj_2024/logs/' + str(n) + 'fold_trainlossL.npy', train_lossL)
    np.save('/mnt/bysj_2024/logs/' + str(n) + 'fold_testaccL.npy', test_accL)
    np.save('/mnt/bysj_2024/logs/' + str(n) + 'fold_testlossL.npy', test_lossL)
    print('save success')
print('10 fold test average accuracy: ', np.mean(save_testacc), '10 fold test average std:', np.std(save_testacc))
print('10 fold test f1 score: ', np.mean(save_testf1), '10 fold test f1 std:', np.std(save_testf1))
np.save('/mnt/bysj_2024/logs/testacc.npy', save_testacc)
np.save('/mnt/bysj_2024/logs/testf1.npy', save_testf1)
np.save('/mnt/bysj_2024/logs/testloss.npy', save_testloss)
'''