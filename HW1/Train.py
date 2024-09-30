import glob
import os
import numpy as np
from sklearn.model_selection import  train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import Fn as Fn
from torch import dropout_, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data_resized/train", help="training data path")
    parser.add_argument("--weights_path", type=str, default="./models", help="path for saving weights")
    parser.add_argument("--CUDA", type=str, default="0", help="trained model path")
    info = parser.parse_args()

    
    #model_type = 'Resnet34GRU'
    model_type = 'Resnet18GRU'
    device = torch.device("cuda:"+info.CUDA if torch.cuda.is_available() else "cpu")
    batch_size = 20
    num_epoch = 200
    lr = 3e-5
    gru_num_layers = 3
    gru_hidden_size = 256
    dropout_p = 0

    n_frames = 16
    #size = 256 
    crop_size = 224
    norm_mean = [0.43216, 0.394666, 0.37645]
    norm_std = [0.22803, 0.22145, 0.216989]



    train_dir = info.data_path
    #test_dir = '/nfs/home/leo0511/CCBDA/HW1/data_resized/test'
    #train_dir = '/nfs/home/leo0511/CCBDA/HW1/data/train'
    #test_dir = '/nfs/home/leo0511/CCBDA/HW1/data/test'
    category_dir_list = os.listdir(train_dir) 
    print('categories amount:',len(category_dir_list))
    train_list = []
    train_label_list = []
    #test_list = []
    for cat in category_dir_list:
        train_list.append(glob.glob(os.path.join(train_dir+'/'+cat,'*.mp4'))) 
        for i in range(len(glob.glob(os.path.join(train_dir+'/'+cat,'*.mp4')))) :
            train_label_list.append(cat)
    #test_list = glob.glob(os.path.join(test_dir,'*.mp4'))
    #print(train_list)
    train_list = [i for item in train_list for i in item]
    print('train data amount:',len(train_list))
    print('label list amount:', len(train_label_list))
    #print(test_list)
    #print('test data amount:',len(test_list))


    # In[2]:



    train_videos, val_videos, train_labels, val_labels = train_test_split(train_list,train_label_list,test_size = 0.25, random_state = 42)
    print("train videos amount:",len(train_videos))
    print("train labels amount:",len(train_labels))
    print("validation videos amount:",len(val_videos))
    print("validation videos amount:",len(val_labels))



    #train_seq_len_list = []
    #val_seq_len_list = []
    train_data = Fn.dataset(train_videos, train_labels,n_frames,crop_size,norm_mean,norm_std)
    val_data = Fn.dataset(val_videos, val_labels, n_frames,crop_size,norm_mean,norm_std)


    print("length of train data:",len(train_data))
    print("length of validation data:",len(val_data))
    '''

    # In[5]:


    #print(train_data[19540])
    imgs, label_seq_len = train_data[19540]
    print(": {}, {}".format(imgs.shape, label_seq_len))
    print('-'*100)
    for i in range(0,20000,2000):
        imgs, label_seq_len = train_data[i]
        print("train data: {}, {}, {}".format(i,imgs.shape, label_seq_len))
    for i in range(0,7000,2000):
        imgs, label_seq_len = val_data[i]
        print("validation data: {}, {}, {}".format(i,imgs.shape, label_seq_len ))
    '''

    # In[6]:



    train_DataLoaded = DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    val_DataLoaded = DataLoader(val_data,batch_size=batch_size*2,shuffle=False,num_workers=16, pin_memory=True, persistent_workers=True)
    print("train data lenth:", len(train_DataLoaded.dataset))
    print("val data lenth:", len(val_DataLoaded.dataset))


    # In[7]:

    if model_type == 'Resnet34GRU':
        model = Fn.Resnet34GRU(num_labels=39,n_frames=n_frames, dropout_p=dropout_p, gru_hidden_size = gru_hidden_size, gru_num_layers=gru_num_layers)
    elif model_type == 'Resnet18GRU':
        model = Fn.Resnet18GRU(num_labels=39,n_frames=n_frames, dropout_p=dropout_p, gru_hidden_size = gru_hidden_size, gru_num_layers=gru_num_layers)
    model.to(device)

    model.load_state_dict(torch.load("/nfs/home/leo0511/CCBDA/HW1/models/3-256_Resnet18GRU_lr3e-05_epoch300_batch20_frames16/best_weights/bestweights.pth",map_location='cuda:'+info.CUDA))

    '''
    for a,b in train_DataLoaded:
        print(a.shape)
        y = model(a,b[1])
        print(y.shape)
        break
    '''

    # In[8]:



    loss_func = nn.CrossEntropyLoss(reduction="sum")
    #opt = optim.Adam(model.parameters(), lr=lr)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=5,verbose=1)
    #model_path = info.weights_path+"/2fc0r_"+f'dp{str(dropout_p)}_gru{str(gru_num_layers)}-{str(gru_hidden_size)}_'+model_type+f'_lr{lr}_epoch{num_epoch}_batch{batch_size}_frames{n_frames}/'
    model_path = info.weights_path+'/'
    os.makedirs(model_path+"checkpoints", exist_ok=True)
    os.makedirs(model_path+"best_weights", exist_ok=True)
    #os.makedirs(model_path+"plot", exist_ok=True)

    print("Using "+model_type+" on ",device)
        

    model,loss_hist,metric_hist = Fn.train_val(
        model=model,
        num_epochs = num_epoch,
        opt = opt,
        loss_fn = loss_func,
        train_data = train_DataLoaded,
        val_data = val_DataLoaded,
        lr_scheduler = lr_scheduler,
        device = device,
        best_weights_path = model_path+"best_weights"+"/bestweights",
        checkpoint_path = model_path+"checkpoints"+"/checkpoint_epoch_",
        model_type=model_type,
        batch_size=batch_size,
        lr=lr,
        n_frames=n_frames,
        gru_num_layers=gru_num_layers,
        gru_hidden_size=gru_hidden_size,
        dropout_p=dropout_p)

    #Fn.plot_loss(loss_hist, metric_hist, plot_path=model_path+"plot"+"/Training_curve.png")


if __name__ == '__main__':
    main()