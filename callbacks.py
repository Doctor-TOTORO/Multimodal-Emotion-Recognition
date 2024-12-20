import datetime
import os
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
class LossHistory():
    def __init__(self,log_dir,model):
        time_str=datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        self.log_dir=os.path.join(log_dir,'loss_'+str(time_str))
        self.losses=[]
        self.val_loss=[]

        os.makedirs(self.log_dir)
        self.writer=SummaryWriter(self.log_dir)


    def append_loss(self,epoch,loss,val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir,'epoch_loss.txt'),'a') as f:
            f.write('Epoch:'+str(epoch)+',')
            f.write('train loss:'+str(loss))
            f.write('\n')
        with open(os.path.join(self.log_dir,'epoch_val_loss.txt'),'a') as f:
            f.write('Epoch:'+str(epoch) + ',')
            f.write('val loss:'+str(val_loss))
            f.write('\n')
        self.writer.add_scalar('loss',loss,epoch)
        self.writer.add_scalar('val_loss',val_loss,epoch)
        self.loss_plot()
    def loss_plot(self):
        iters=range(len(self.losses))
        plt.figure()
        plt.plot(iters,self.losses,'red',linewidth=2,label='train loss')
        plt.plot(iters,self.val_loss,'coral',linewidth=2,label='test loss')
        try:
            if len(self.losses)<25:
                num=5
            else:
                num=15
                # 损失平滑处理
                plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                         label='smooth train loss')
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--',
                         linewidth=2,
                         label='smooth test loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        plt.savefig(os.path.join(self.log_dir,'epoch_loss.png'))
        plt.cla()
        plt.close('all')