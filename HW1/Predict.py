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
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import csv
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data_resized/test", help="testing data path")
    parser.add_argument("--weights_path", type=str, default="./models/best_weights/bestweights.pth", help="weights path")
    parser.add_argument("--prediction_path", type=str, default="./", help="path for saving prediction")
    parser.add_argument("--CUDA", type=str, default="0", help="trained model path")
    info = parser.parse_args()

    batch_size = 20
    gru_num_layers = 3
    gru_hidden_size = 256
    dropout_p = 0

    gpu_num=int(info.CUDA)
    #n_th = 0
    weights_path = info.weights_path
    weights_name = os.path.split(os.path.split(os.path.split(weights_path)[0])[0])[1].strip('.pth')
    n_frames = 16
    

    #model_type = 'Resnet34GRU'
    model_type = 'Resnet18GRU'
    #gpu_num = 0
    device = torch.device("cuda:"+info.CUDA)
  



    #size = 256
    crop_size = 224
    norm_mean = [0.43216, 0.394666, 0.37645]
    norm_std = [0.22803, 0.22145, 0.216989]

    test_dir = info.data_path
    test_videos = os.listdir(test_dir) 
    test_list = []
    print('video amount:',len(test_videos))
    for v in test_videos:
        test_list.append(test_dir+'/'+v)
    #print(test_list) 
        
    print("test data amount: ",len(test_list))

    test_data = Fn.test_dataset(test_list,n_frames,crop_size,norm_mean,norm_std)
    print("length of test data:",len(test_data))

    test_DataLoaded = DataLoader(test_data, batch_size=batch_size,shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
    print("train data lenth:", len(test_DataLoaded.dataset))

    print("using model type: ",model_type)
    if model_type == 'Resnet34GRU':
        model = Fn.Resnet34GRU(num_labels=39,n_frames=n_frames, dropout_p=dropout_p, gru_hidden_size=gru_hidden_size, gru_num_layers=gru_num_layers)
    elif model_type == 'Resnet18GRU':
        model = Fn.Resnet18GRU(num_labels=39,n_frames=n_frames, dropout_p=dropout_p, gru_hidden_size=gru_hidden_size, gru_num_layers=gru_num_layers)
    model.to(device)
    print("loading weights name: ",weights_name)
    model.load_state_dict(torch.load(weights_path,map_location='cuda:'+info.CUDA))



    print("start prediction...")
    pred = Fn.pedict(model=model, 
            test_data=test_DataLoaded,
            device=device,
            gpu_num=gpu_num,
            model_type=model_type,
            batch_size=batch_size,
            n_frames=n_frames)
    print(type(pred))
    print(np.shape(pred))


    pred_path = info.prediction_path
    #os.makedirs(pred_path,exist_ok=True)
    print("saving pridiction in: ",pred_path)


    with open(pred_path + f'prediction.csv', 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(['name','label'])
      table = []
      for i in range(10000):
        
        name = os.path.split(test_list[i])[1]
        answer = pred[i]
        table.append([name,str(answer)])

        width = 40
        progress = (i/10000)
        xlen = int(progress*width)
        dlen = width - xlen
        print(f"[{'=' * xlen}{'.' * dlen}] {progress * 100:.1f}%",end='\r', flush=True)
      writer.writerows(table)
      print("Done!!")


if __name__ == '__main__':
    main()