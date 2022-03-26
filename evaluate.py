'''The following module trains the weights of the neural network model.'''
import os
import datetime
import uuid
from tqdm import tqdm

import torch
import torch.nn    as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from dataloader import Mydataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tab import TabTransformer
from loader_helper        import LoaderHelper
from   sklearn.metrics   import auc
import matplotlib.pyplot as plt
import numpy as np

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print("Running on the GPU.")
else:
    DEVICE = torch.device("cpu")
    print("Running on the CPU")




    

def get_roc_auc(model_in, test_dl, figure=False, path=None, fold=1):
    
    fpr = [] #1-specificity
    tpr = []

    youden_s_lst = []

    opt_acc = 0; opt_sens = 0; opt_spec = 0
    youdens_s_max = 0
    optimal_thresh = 0
    labels = ['False', 'True']
    print("Walking through thresholds.")
    for t in range(0, 10, 1):

        thresh = t/10
        acc, sens, spec, cm = get_metrics(model_in, test_dl, thresh)
        tpr.append(sens)
        fpr.append(1 - spec)
        

        youdens_s = sens + spec - 1

        if (youdens_s > youdens_s_max): 

            youdens_s_max = youdens_s; 
            optimal_thresh = thresh
            opt_acc = acc; opt_sens = sens; opt_spec = spec; opt_cm = cm

        
        

    roc_auc = -1
    try:
        roc_auc = auc(fpr, tpr)
    except Exception as e:
        print(e)
    metrics = [opt_acc, opt_sens, opt_spec, roc_auc, youdens_s_max, optimal_thresh]
    
    
    print("ACC = {}, BACC = {}, AUC = {}, SENS = {}, SPEC={}".format(opt_acc,(opt_sens+opt_spec)/2,roc_auc,opt_sens,opt_spec))
    disp = ConfusionMatrixDisplay(confusion_matrix=opt_cm, display_labels=labels)
    disp.plot()
    plt.savefig(path + "/confusionMatrix{}.png".format(fold))
    if(figure):

        if (path == None):
            path = "../graphs/auc-{date:%Y-%m-%d_%H-%M-%S}.png".format(date=datetime.datetime.now())
        else:
            #append dir
            path = path + "/auc-fold{}-{date:%Y-%m-%d_%H-%M-%S}.png".format(fold, date=datetime.datetime.now())
        
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Fold {}'.format(fold))
        plt.legend(loc="lower right")
        plt.savefig(path)
    
    return metrics


def get_metrics(model_in, test_dl, thresh=0.5, param_count=False):
        
    correct = 0; total = 0
    model_in.eval()
    
    TP = 0.000001; TN = 0.000001; FP = 0.000001; FN = 0.000001
    
    with torch.no_grad():
        
        for i_batch, sample_batched in enumerate(test_dl):
            
            batch_X, batch_y  = sample_batched
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.cpu()
            net_out = model_in(batch_X).cpu()

            net_out = np.array(net_out)
            net_out[net_out>thresh] = 1
            net_out[net_out<thresh] = 0
            
            cm = confusion_matrix(batch_y, net_out)
            zipped = zip((TN, FP, FN, TP), cm.ravel()) #使用zip方法进行连接
            mapped = map(sum, zipped) #使用sum进行求和计算，map方法映射
            TN, FP, FN, TP = tuple(mapped)
            

            
    
    accuracy = round((TP + TN)/(TN + FP + FN + TP), 4)
    sensitivity = round((TP / (TP + FN)), 4)
    specificity = round((TN / (TN + FP)), 4)

    
    
    return (accuracy, sensitivity, specificity, cm)
    
def load_cam_model(path):
    model = torch.load(path)
    return model

def evaluate_model( k_folds=5):
    '''The function for training the camull network'''
    
    
    ld_helper = LoaderHelper()
    uuid = "Tab_2022-03-25_203638"
    
    for k_ind in range(k_folds):
        path = "../weights/"+uuid+"/best_weight_fold_{}".format(k_ind+1)
        model   = load_cam_model(path)
        test_data = ld_helper.get_test_dl(k_ind)
        if (not os.path.exists("../graphs/" + uuid)) : os.mkdir("../graphs/" + uuid)
        metrics = get_roc_auc(model, test_data, figure=True, path = "../graphs/" + uuid, fold=k_ind+1)
        
        




def main():
    '''Main function of the module.'''
    #NC v AD
    

    
    evaluate_model()


    #ld_helper = LoaderHelper(task=Task.sMCI_v_pMCI)
    
    #model_uuid = "Densenet_2022-01-22_14:36:19"
    #evaluate_model(DEVICE, model_uuid, ld_helper)

    #transfer learning for pMCI v sMCI
    #ld_helper.change_task(Task.sMCI_v_pMCI)
    #uuid = "25d4d977962d4929b0553d7e6ec9c049"
    # model = load_model()
    # uuid  = train_camull(ld_helper, model=model, epochs=60)
    # uuid = train_camull(ld_helper, epochs=20)
    # evaluate_model(DEVICE, uuid, ld_helper)

main()
