import os
import torch
import numpy as np
class EarlyStopping:
    def __init__(self,save_path,index=0,patience=20,verbose=True,delta=0):
        '''

        :param save_path: saving folders
        :param patience: stop counts
        :param verbose: if True, print a message for each validation
        :param delta: monitor the variation, default is 0
        '''
        self.save_path=save_path
        self.patience=patience
        self.verbose=verbose
        self.counter=0
        self.best_score=None
        self.early_stop=False
        self.val_loss_min=np.Inf
        self.delta=delta
        self.index=index

    def __call__(self, val_loss, model):
        score=-val_loss
        if self.best_score is None:
            self.best_score=score
            self.save_checkpoint(val_loss,model)
        elif score<self.best_score+self.delta:
            self.counter+=1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop=True
        else:
            self.best_score=score
            self.save_checkpoint(val_loss,model)
            self.counter=0

    def save_checkpoint(self,val_loss,model):
        """
        save model when the loss is decreasing
        :param val_loss:
        :param model:
        :return:
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        #bestmodel_path=os.path.join('/root/autodl-tmp/WESAD_ECG_3class_10s/logs/model_save','best_network.pth')
        bestmodel_path='/mnt/bysj_2024/logs/model_save/best_model.pth'
        torch.save(model.state_dict(),bestmodel_path)
        self.val_loss_min=val_loss

