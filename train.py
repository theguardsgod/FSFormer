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

from tab import TabTransformer
from loader_helper        import LoaderHelper

import matplotlib.pyplot as plt

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print("Running on the GPU.")
else:
    DEVICE = torch.device("cpu")
    print("Running on the CPU")

def save_weights(model_in, uuid_arg, epoch, fold=1):
    '''The following function saves the weights file into required folder'''
    
    
    root_path = "../weights/"     + uuid_arg + "/"
    
        

    if os.path.exists(root_path) == False:
        os.mkdir(root_path) #otherwise it already exists

    while True:

        s_path = root_path + "fold_{}_epoch{}_weights-{date:%Y-%m-%d_%H%M%S}".format(fold, epoch, date=datetime.datetime.now()) # pylint: disable=line-too-long

        if os.path.exists(s_path):
            print("Path exists. Choosing another path.")
        else:
            torch.save(model_in, s_path)
            break
def save_best_weights(model_in, uuid_arg,fold=1):
    '''The following function saves the weights file into required folder'''
    root_path = "../weights/"     + uuid_arg + "/"

    if os.path.exists(root_path) == False:
        os.mkdir(root_path) #otherwise it already exists

    

    s_path = root_path + "best_weight_fold_{}".format(fold) # pylint: disable=line-too-long

        
    torch.save(model_in, s_path)
   

def load_cam_model(path):
    model = torch.load(path)
    return model

def load_model():
    '''Function for loaded camull net from a specified weights path'''
    
    model = load_cam_model("../weights/NC_v_AD/DensenetCrossAttention_2022-02-20_13:43:19/best_weight")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model.to(DEVICE)
    #model = nn.DataParallel(model)
    
    

    return model

def build_arch(num_features):
    '''Function for instantiating the pytorch neural network object'''
    net = TabTransformer(
            num_features = num_features,
            dim = 64,                           # dimension, paper set at 32
                       # binary prediction, but could be anything
            depth = 6,                          # depth, paper recommended 6
            heads = 8,                          # heads, paper recommends 8
            attn_dropout = 0.0,                 # post-attention dropout
            ff_dropout = 0.0,                   # feed forward dropout

            mlp_act = nn.ReLU()               # activation for final mlp, defaults to relu, but could be anything else (selu etc)
            
        )
    # net = HeterogeneousResNetwithClinical()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    net.to(DEVICE)
    net.double()

    return net



# def evaluate(model_in, test_dl, thresh=0.5, param_count=False):
        
#     correct = 0; total = 0
#     model_in.eval()
    
#     TP = 0.000001; TN = 0.000001; FP = 0.000001; FN = 0.000001
    
#     with torch.no_grad():
        
#         for i_batch, sample_batched in enumerate(test_dl):
            
#             batch_X  = sample_batched['mri'].to(DEVICE)
#             # batch_clinical = sample_batched['clin_t'].to(DEVICE)
#             batch_y  = sample_batched['label'].to(DEVICE)
            
#             for i in range(len(batch_X)): #hard coded batch size of 4
                
#                 real_class = batch_y[i].item()
#                 X = batch_X[i].unsqueeze(1)
#                 # clinical = batch_clinical[i].unsqueeze(0)
                
#                 # net_out = model_in(X,clinical)
#                 net_out = model_in(X)
#                 predicted_class = 1 if net_out > thresh else 0
                
#                 if (predicted_class == real_class):
#                     correct += 1
#                     if (real_class == 0):
#                         TN += 1
#                     elif (real_class == 1):
#                         TP += 1
#                 else:
#                     if (real_class == 0):
#                         FP += 1
#                     elif (real_class == 1):
#                         FN += 1
                    
                    
#                 total += 1

            
    
#     accuracy = round(correct/total, 5)
#     sensitivity = round((TP / (TP + FN)), 5)
#     specificity = round((TN / (TN + FP)), 5)

    
    
#     return (accuracy, sensitivity, specificity)

def evaluate(model_in, test_dl, thresh=0.5, param_count=False):
        
    correct = 0; total = 0
    model_in.eval()
    
    TP = 0.000001; TN = 0.000001; FP = 0.000001; FN = 0.000001
    
    with torch.no_grad():
        
        for i_batch, sample_batched in enumerate(test_dl):
            
            batch_X, batch_y  = sample_batched
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            

            net_out = model_in(batch_X)
            


            for i in range(len(batch_X)): #hard coded batch size of 4
                
                real_class = batch_y[i].item()
                
                predicted_class = 1 if net_out[i] > thresh else 0
                
                
                
                if (predicted_class == real_class):
                    correct += 1
                    if (real_class == 0):
                        TN += 1
                    elif (real_class == 1):
                        TP += 1
                else:
                    if (real_class == 0):
                        FP += 1
                    elif (real_class == 1):
                        FN += 1
                    
                    
                total += 1

            
    
    sensitivity = round((TP / (TP + FN)), 5)
    specificity = round((TN / (TN + FP)), 5)
    accuracy = round((sensitivity+specificity)/2, 5)

    
    
    return accuracy, sensitivity, specificity

def train_loop(model_in, train_dl, test_dl, epochs, uuid_, k_folds):
    '''Function containing the neural net model training loop'''
    optimizer = optim.AdamW(model_in.parameters(), lr=0.0001, weight_decay=5e-9)
    #scheduler_warm = lr_scheduler.ConstantLR(optimizer, start_factor=0.2, total_iters=5)
    scheduler_warm = lr_scheduler.StepLR(optimizer,step_size=1, gamma=1.4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=1)
    loss_function = nn.BCELoss()
    loss_fig = []
    eva_fig = []

    model_in.train()
    best_acc = 0
    nb_batch = len(train_dl)
    log_path = "../train_log/" + uuid_ +".txt"

    if (os.path.exists(log_path)):
        filein     = open(log_path, 'a')
    else:
        filein     = open(log_path, 'w')
    print("k = {}".format(k_folds))
    fig1 = plt.figure(1)
    # Train
    for i in range(epochs):
        loss = 0.0
        model_in.train()
        for _, sample_batched in enumerate(tqdm(train_dl)):

            batch_X, batch_y  = sample_batched
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            model_in.zero_grad()
            outputs = model_in(batch_X)
            
            
            batch_loss = loss_function(outputs, batch_y)
            batch_loss.backward()
            optimizer.step()
            loss += float(batch_loss) / nb_batch

        tqdm.write("Epoch: {}/{}, train loss: {}".format(i, epochs, round(loss, 5)))
        filein.write("Epoch: {}/{}, train loss: {}\n".format(i, epochs, round(loss, 5)))
        loss_fig.append(round(loss, 5))
        accuracy, sensitivity, specificity = evaluate(model_in, test_dl)
        eva_fig.append(accuracy)
        tqdm.write("Epoch: {}/{}, evaluation loss: {}".format(i, epochs,(accuracy, sensitivity, specificity)))
        filein.write("Epoch: {}/{}, evaluation loss: {}\n".format(i, epochs,(accuracy, sensitivity, specificity)))
        if i % 10 == 0 and i != 0:
            save_weights(model_in, uuid_, epoch = i, fold=k_folds)
            plt.plot(range(i+1),loss_fig,ls="-", lw=2,label="training loss")
            plt.plot(range(i+1),eva_fig,ls="-", lw=2,label="evaluation loss")
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training curve - Fold {}'.format(k_folds))
            plt.savefig("../figures/"+uuid_+'/'+str(k_folds) + 'eva.png')        

        if accuracy >= best_acc:
           save_best_weights(model_in, uuid_,k_folds)
           best_acc = accuracy

        if epochs <= 5:
            scheduler_warm.step() 
        else:            
            scheduler.step(loss)
        
    plt.close(fig1)
    
    


def train_camull( k_folds=5, model=None, epochs=40):
    '''The function for training the camull network'''
    
    uuid_ = "TabIntrusmote_{date:%Y-%m-%d_%H%M%S}".format(date=datetime.datetime.now())
    ld_helper = LoaderHelper()
    model_cop = model
    datasetName = "intrusmote"
    num_features = 71
    os.mkdir("../figures/"+uuid_+'/')
    for k_ind in range(k_folds):

        if model_cop is None:
            model = build_arch(num_features=num_features)
        else:
            model = model_cop
        
        
        train_data = ld_helper.get_train_dl(datasetName,k_ind+1,num_features)
        test_data = ld_helper.get_test_dl(datasetName,k_ind+1,num_features)
        train_loop(model, train_data, test_data, epochs, uuid_, k_folds=k_ind+1)
        

        

        print("Completed fold {}/{}.".format(k_ind, k_folds))

    return uuid_





def main():
    '''Main function of the module.'''
    #NC v AD
    

    
    model_uuid = train_camull(epochs=80)


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
