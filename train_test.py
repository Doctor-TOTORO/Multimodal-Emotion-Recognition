import torch
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def test_one_epoch(model, criterion, data_loader, test_num,final_test,k):
    running_loss = 0.0
    correct_num = 0
    all_true_label = []
    all_pred_label = []
    model.eval()
    batch_size = None
    with tqdm(enumerate(data_loader), total=len(data_loader),ncols=85) as tbar:
        for index, (x, y) in tbar:
            batch_size = x.shape[0] if index == 0 else batch_size
            x = x.cuda().float().contiguous()
            y = y.cuda().long().contiguous()
            out_e = model(x)
            _, pred = torch.max(out_e, 1)
            correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
            loss = criterion[0](out_e, y.long())
            running_loss += float(loss.item())
            all_true_label.append(y.cpu())
            all_pred_label.append(pred.cpu())

    batch_num = test_num // batch_size
    _loss = running_loss / (batch_num + 1)
    acc = correct_num / test_num * 100

    # f1 score
    all_true_label = torch.cat(all_true_label, dim=0)
    all_pred_label = torch.cat(all_pred_label, dim=0)
    f1 = metrics.f1_score(all_true_label, all_pred_label, average='macro')
    f1=f1*100
    if final_test==True:
        true_labels = all_true_label  # real label
        # print('2', f_y_true)
        predicted_labels = all_pred_label  # prediction label
        # confusion matrix
        confusion = confusion_matrix(true_labels, predicted_labels)
        print(confusion)
        confusion_prob = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
        # create image of confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_prob, interpolation='nearest', cmap=plt.cm.YlGnBu)  # YlGnBu mapping
        plt.colorbar()
        plt.title('Confusion Matrix', fontsize=20)

        plt.xticks(np.arange(3), labels=['0', '1', '2'], fontsize=20)
        plt.yticks(np.arange(3), labels=['0', '1', '2'], fontsize=20)

        plt.xlabel('Predicted', fontsize=20)
        plt.ylabel('True', fontsize=20)
        for i in range(3):
            for j in range(3):
                plt.text(j, i, format(confusion_prob[i, j], ".2f"), ha="center", va="center", color="black", weight="bold")
        # save as .png
        plt.savefig('/mnt/pycharm_ftq/CSI/confusion_matrix/confusion_matrix' + str(k) + '.png', dpi=600)
    print(f'Test loss: {_loss:.4f}\tTest acc: {acc:.2f}%\tF1: {f1:.4f}')
    return _loss,acc,f1



def train_one_epoch(model, criterion, optimizer, data_loader, train_num, epoch, epochs,test_loader, test_num, train_accL,
                    train_lossL, test_accL, test_lossL, f1L, device, max_testacc, max_f1, best_test_loss,early_stopping,
                    loss_history):
    model.train()
    running_loss = 0.0
    correct_num = 0

    batch_size = None
    with tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch{epoch + 1}/{epochs}', unit='it',ncols=85) as tbar:
        for index, (x, y) in tbar:
            batch_size = x.shape[0]//2 if index == 0 else batch_size
            x = x.cuda().float().contiguous()
            y = y.cuda().long().contiguous()
            # y=y.reshape(-1,1)
            out_e = model(x)
            _, pred = torch.max(out_e, 1)
            correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
            loss= criterion[0](out_e, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        batch_num = train_num // batch_size
        _loss = running_loss / (batch_num + 1)
        acc = correct_num / train_num * 100
        print("Test begin...")
        test_loss, test_acc,f1= test_one_epoch(model, criterion, test_loader, test_num,final_test=False,k=None)
        print(
            f'Epoch {epoch + 1}/{epochs}\tTrain loss: {_loss:.4f}\tTrain acc: {acc:.2f}%\tTest loss: {test_loss:.4f}\tTest acc: {test_acc:.2f}%\tf1score: {f1:.4f}')
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            print('best test_loss decreased to %.4f' % best_test_loss)
        early_stopping(test_loss, model)
        es = early_stopping.early_stop
        min_loss = early_stopping.val_loss_min
        if test_acc > max_testacc:
            max_testacc = test_acc
            max_f1 = f1
        loss_history.append_loss(epoch + 1, _loss, test_loss)
        train_accL.append(acc)
        train_lossL.append(_loss)
        test_accL.append(test_acc)
        test_lossL.append(test_loss)
        f1L.append(f1)
        return train_accL, train_lossL, test_accL, test_lossL, f1L, min_loss, es, max_testacc, max_f1, best_test_loss
# ----------------------------------------------------------------------------------------------#
##
