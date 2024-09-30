#from turtle import forward
#from pyexpat import model
from csv import writer
import cv2
import numpy as np
import PIL.Image as Image
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import to_pil_image
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter  
#from tqdm import tqdm_notebook




log_writer = SummaryWriter()
#########################################################
#                       dataset                         #
#########################################################




class dataset(Dataset):
    def __init__(self, data_list, label_list,n_frames,crop_size, norm_mean, norm_std):
        self.data_list = data_list
        self.label_list = label_list
        self.n_frames = n_frames
        self.crop_size = crop_size 
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        

    def __getitem__(self, index ):
        self.transformed_frames = []
        video_path = self.data_list[index]
        label = int(self.label_list[index])
        #print(type(label))
        #print("-"*100+"dataset getitem start"+"-"*100)
        frames, length, width, height = get_frames(video_path, self.n_frames) #從影片中取subsample = 16 frames
        #print("-"*100+"dataset transform start"+"-"*100)
        
        for f in range(len(frames)):
            self.transformed_frames.append(tansform_frames(frames[f],self.crop_size, self.norm_mean, self.norm_std)) #前處理每個frame
        if len(self.transformed_frames)>0:
            self.transformed_frames = torch.stack(self.transformed_frames)
        self.seq_len = len(self.transformed_frames)
        #print("-"*100+"dataset padding start"+"-"*100)
        if self.seq_len < self.n_frames:
            self.padded_frames = padding_frame(self.transformed_frames,self.n_frames,width=self.crop_size,height=self.crop_size)
        else:
            self.padded_frames = self.transformed_frames
        label_seq_len = [label,self.seq_len]
        #print("-"*100+"dataset getitem end"+"-"*100)
        
        
        return self.padded_frames, label_seq_len
    
    def __len__(self):

        return len(self.data_list)
        
        
class test_dataset(Dataset):
    def __init__(self, data_list, n_frames,crop_size, norm_mean, norm_std):
        self.data_list = data_list
        self.n_frames = n_frames
        self.crop_size = crop_size 
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        

    def __getitem__(self, index ):
        self.transformed_frames = []
        video_path = self.data_list[index]
        #print(type(label))
        #print("-"*100+"dataset getitem start"+"-"*100)
        frames, length, width, height = get_frames(video_path, self.n_frames) #從影片中取subsample = n_frames
        #print("frames: ",np.shape(frames))
        #print("-"*100+"dataset transform start"+"-"*100)
        for f in range(len(frames)):
            self.transformed_frames.append(tansform_frames(frames[f],self.crop_size, self.norm_mean, self.norm_std)) #前處理每個frame
        if len(self.transformed_frames)>0:
            self.transformed_frames = torch.stack(self.transformed_frames)
        #print("transformed: ",self.transformed_frames)
        self.seq_len = len(self.transformed_frames)
        #print("-"*100+"dataset padding start"+"-"*100)
        if self.seq_len < self.n_frames:
            self.padded_frames = padding_frame(self.transformed_frames,self.n_frames,width=self.crop_size,height=self.crop_size)
        else:
            self.padded_frames = self.transformed_frames
        #print("padded: ",self.padded_frames)
        #print("-"*100+"dataset getitem end"+"-"*100)
        
        
        return self.padded_frames, self.seq_len
    
    def __len__(self):

        return len(self.data_list)


#########################################################
#                     preprocessing                     #
#########################################################

def get_frames(filename,n_frames=16):
    n_frames = n_frames-1
    frames = []
    cap = cv2.VideoCapture(filename)
    width =int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list= np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    for i in range(v_len):
        ret, frame = cap.read()
        if not ret:
            break
        if (i in frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            frames.append(frame)
    cap.release()
    return frames, v_len, width, height


def tansform_frames(frame,crop_size, norm_mean, norm_std ):
    #print(type(frame))
    frame = Image.fromarray(np.array(frame))
    #print(type(frame))
    transform = transforms.Compose([

        #transforms.Resize((256,256)),

        transforms.CenterCrop((crop_size,crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean = norm_mean,std=norm_std)
    ])
    transformed_frame = transform(frame)
    return transformed_frame

def denormalize(x_, mean, std):
    x = x_.clone()
    for i in range(3):
        x[i] = x[i]*std[i]+mean[i]
    x = to_pil_image(x)        
    return x

def padding_frame(frame,seq_len,width,height):
    refference = torch.ones(seq_len,3,width,height) 

    padded = pad_sequence([frame,refference],batch_first = True,padding_value = 0.)
    padded_frame = padded[0]
    
    return padded_frame

#########################################################
#                         model                         #
#########################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import models
# Create CNN Model



class GRUNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_labels,n_frames, dropout_p):
        super(GRUNet, self).__init__()
        self.gru1 = nn.GRU(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers, batch_first = True)
        #self.fc0 = nn.Linear(hidden_size, int(hidden_size/2))
        self.fc1 = nn.Linear(int(hidden_size),num_labels)
        #self.batch_norm1  = nn.BatchNorm1d(hidden_size*n_frames )
        self.dropout = nn.Dropout1d(dropout_p)

    def forward(self,x):#input( batch,seq_len, input_size)
        #print("input :",data.shape)
        #print(data)
        x, _ = self.gru1(x) 
        #assert torch.isnan(out).sum() == 0, print("gru1",out)
        #print("packed after gru: ",x[0].shape,x[1].shape)
        unpacked_seq, unpacked_lens = pad_packed_sequence(x, batch_first=True) 
        x = unpacked_seq[:,-1] # 取最後一個輸出(Batch_size, hidden_size)
        #x = unpacked_seq.view(unpacked_seq.shape[0], -1) # out: (Batch_size, n_frames*hidden_size)
        #x = self.batch_norm1(x)
        #x = self.fc0(x)
        #x = F.relu(x)

        #x = self.dropout(x)
        
        #print("unpack: ",unpacked_seq.shape)
        #print("output of LSTM :",out.shape)
        #x = self.hidden2tag(x[:,-1])
        x = self.fc1(x) 
        #assert torch.isnan(out).sum() == 0, print("hidden2tag",out)
        #print("output of Linear :",x.shape)
        x = F.softmax(x, dim=1)
        #assert torch.isnan(out).sum() == 0, print("softmax",out)
        #print("output of softmax :",out.shape)
        return x
'''
class LSTMNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_labels,n_frames, dropout_p):
        super(LSTMNet, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers, batch_first = True)
        self.fc1 = nn.Linear(hidden_size,num_labels)
        #self.batch_norm1  = nn.BatchNorm1d(hidden_size*n_frames )
        #self.dropout = nn.Dropout1d(dropout_p)

    def forward(self,x):#input( batch,seq_len, input_size)
        #print("input :",data.shape)
        #print(data)
        x, _ = self.lstm1(x) 
        #assert torch.isnan(out).sum() == 0, print("gru1",out)
        #print("packed after gru: ",x[0].shape,x[1].shape)
        unpacked_seq, unpacked_lens = pad_packed_sequence(x, batch_first=True) 
        x = unpacked_seq[:,-1] # 取最後一個輸出(Batch_size, hidden_size)
        #x = unpacked_seq.view(unpacked_seq.shape[0], -1) # out: (Batch_size, n_frames*hidden_size)
        #x = self.batch_norm1(x)
        #x = self.dropout(x)
        #print("unpack: ",unpacked_seq.shape)
        #print("output of LSTM :",out.shape)
        #x = self.hidden2tag(x[:,-1])
        x = self.fc1(x) 
        #assert torch.isnan(out).sum() == 0, print("hidden2tag",out)
        #print("output of Linear :",x.shape)
        x = F.softmax(x, dim=1)
        #assert torch.isnan(out).sum() == 0, print("softmax",out)
        #print("output of softmax :",out.shape)
        return x
'''

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x    
   
class Resnet18GRU(nn.Module):
    def __init__(self, num_labels, n_frames, dropout_p, gru_hidden_size, gru_num_layers):
        super(Resnet18GRU,self).__init__()
        self.resnet18 =  models.resnet18(pretrained = False)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = Identity()
        self.gru = GRUNet(input_size=num_features, hidden_size=gru_hidden_size, num_layers=gru_num_layers, num_labels= num_labels , n_frames = n_frames, dropout_p = dropout_p)
        
    
    def forward(self, x, seq_len_list): #input size : (batch_size, n_frames, channels, H, W)
        #print("-"*100+"model computing (resnet18) start"+"-"*100)
        #print(type(x))
        batch_size = x.shape[0]
        n_frames = x.shape[1]
        x  = x.view(-1, x.shape[2], x.shape[3], x.shape[4]) # flatten to (Batch_size*n_frames, channels, H, W)
        #print("flatten input:",x.shape)
        x = self.resnet18(x) #resnet18 output: (batch_size*n_frames, 512)
        #x = self.batch_norm1(x) #batch normalization
        #print("output of resnet: ",x.shape)
        x = x.view(batch_size, n_frames, -1) # (Batch_size*n_frames, 512)--> (Batch_size, n_frames, 512)
        #print("unflatten: ",x.shape)
        n_features = x.shape[2] # 512
        #print("-"*100+"pack padded start"+"-"*100)
        x = pack_padded_sequence(x, seq_len_list.cpu(), batch_first=True, enforce_sorted = False)
        #seq_len_list.to(device)
         #pack x with seq_len_list of the batch
        #print("packed: ",x[0].shape,x[1].shape)
        #lstm = LSTM(input_size=n_features,hidden_size=34,n_frames=n_frames)
        #out = lstm(packed_feature)\
        #print("-"*100+"gru start"+"-"*100)
        out = self.gru(x) # output: (Batch_size, num_labels)
        #print(out)
        return out

class Resnet34GRU(nn.Module):
    def __init__(self, num_labels, n_frames, dropout_p, gru_hidden_size, gru_num_layers):
        super(Resnet34GRU,self).__init__()
        self.resnet34 =  models.resnet34(pretrained = False)
        num_features = self.resnet34.fc.in_features
        self.resnet34.fc = Identity()
        self.gru = GRUNet(input_size=num_features, hidden_size=gru_hidden_size, num_layers=gru_num_layers, num_labels= num_labels , n_frames = n_frames, dropout_p = dropout_p)
        
    
    def forward(self, x, seq_len_list): #input size : (batch_size, n_frames, channels, H, W)
        #print("-"*100+"model computing (resnet34) start"+"-"*100)
        batch_size = x.shape[0]
        n_frames = x.shape[1]
        x  = x.view(-1, x.shape[2], x.shape[3], x.shape[4]) # flatten to (Batch_size*n_frames, channels, H, W)
        #print("flatten input:",x.shape)
        x = self.resnet34(x) #resnet34 output: (batch_size*n_frames, 2048)
        #x = self.batch_norm1(x) #batch normalization
        #print("output of resnet: ",x.shape)
        x = x.view(batch_size, n_frames, -1) # (Batch_size*n_frames, 512)--> (Batch_size, n_frames, 2048)
        #print("unflatten: ",x.shape)
        n_features = x.shape[2] # 512
        #print("-"*100+"pack padded start"+"-"*100)
        x = pack_padded_sequence(x, seq_len_list.cpu(), batch_first=True, enforce_sorted = False)
        #seq_len_list.to(device)
         #pack x with seq_len_list of the batch
        #print("packed: ",x[0].shape,x[1].shape)
        #lstm = LSTM(input_size=n_features,hidden_size=50,n_frames=n_frames)
        #out = lstm(packed_feature)\
        #print("-"*100+"gru start"+"-"*100)
        out = self.gru(x) # output: (Batch_size, num_labels)
        #print(out)
        return out


#########################################################
#                         train                         #
#########################################################
import copy


def compute_loss(model,loss_fn,data,device,opt=None):
    running_loss=0.0
    running_metric=0.0
    len_data = len(data.dataset)
    batch = 0
    #print("-"*100+"start compute_loss"+"-"*100)
    for xb, yb in data:
        #print("-"*100+"start a batch"+"-"*100)
        train_batch = xb
        label = yb[0]
        seq_len_list = yb[1]

        #print(seq_len_list)
        train_batch=train_batch.to(device)
        label=label.to(device)
        seq_len_list=seq_len_list.to(device)

        output=model(train_batch,seq_len_list)
        #print("-"*100+"loss computing start"+"-"*100)
        loss_b,metric_b= loss_batch(loss_fn, output, label, opt)
        running_loss+=loss_b

        batch+=1
        width = 40
        progress = (batch/int(len_data/len(xb)))
        xlen = int(progress*width)
        dlen = width - xlen
        print(f"[{'=' * xlen}{'.' * dlen}] {progress * 100:.1f}%",end='\r', flush=True)

        if metric_b is not None:
            running_metric+=metric_b
    print("\n")
    loss=running_loss/float(len_data)
    metric=running_metric/float(len_data)
    return loss, metric


def loss_batch(loss_fn, output, target, opt=None):
    loss = loss_fn(output, target)
    with torch.no_grad():
        pred = output.argmax(dim=1, keepdim=True)
        metric_b=pred.eq(target.view_as(pred)).sum().item()
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def train_val(model, 
    num_epochs,
    loss_fn,
    opt,
    train_data,
    val_data,
    lr_scheduler,
    device,
    best_weights_path,
    checkpoint_path,
    model_type,
    batch_size,
    lr,
    n_frames,
    gru_num_layers,
    gru_hidden_size,
    dropout_p):
    loss_history={
        "train": [],
        "val": [],
    }
    
    metric_history={
        "train": [],
        "val": [],
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=float('inf')
    
    for epoch in range(num_epochs):
        #print("-"*100+"start an epoch"+"-"*100)
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
        model.train()
        print("Training...\n")
        train_loss, train_metric=compute_loss(model,loss_fn,train_data,device,opt)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        model.eval()
        with torch.no_grad():
            print("validating...\n")
            val_loss, val_metric=compute_loss(model,loss_fn,val_data,device)
            

        if epoch%5 ==0:
            torch.save(model.state_dict(), checkpoint_path+str(epoch)+"pth")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), best_weights_path+".pth")
            print("Copied best model weights!")
        
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)
        
        log_writer.add_scalars(main_tag="final_1fc0_"+str(dropout_p)+'dp'+str(gru_num_layers)+'-'+str(gru_hidden_size)+"GRU"+model_type+f'_batch({batch_size})_lr({lr})_frames({n_frames})/loss', 
        tag_scalar_dict={'train_loss':float(train_loss),'val loss':float(val_loss)}, global_step=epoch)
        log_writer.add_scalars(main_tag="final_1fc0_"+str(dropout_p)+'dp'+str(gru_num_layers)+'-'+str(gru_hidden_size)+"GRU"+model_type+f'_batch({batch_size})_lr({lr})_frames({n_frames})/accuracy',
        tag_scalar_dict={'train acc': float(train_metric),'val acc':float(val_metric)},global_step=epoch)
        log_writer.add_scalars(main_tag="final_1fc0_"+str(dropout_p)+'dp'+str(gru_num_layers)+'-'+str(gru_hidden_size)+"GRU"+model_type+f'_batch({batch_size})_lr({lr})_frames({n_frames})/lr',
        tag_scalar_dict={'lr': current_lr},global_step=epoch)


        print("train loss: %.6f, train accuracy: %.2f, val loss: %.6f, val accuracy: %.2f" %(train_loss,100*train_metric,val_loss,100*val_metric))
        print("-"*10) 
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), best_weights_path+"_final.pth")
    writer.close()
    return model, loss_history, metric_history

def pedict(model, 
    test_data,
    device,
    gpu_num,
    model_type,
    batch_size,
    n_frames):
    
    pred_history = []
    model.eval()
    batch = 0
    len_data = len(test_data.dataset)
    with torch.no_grad():
        for xb, yb in test_data:
            test_batch = xb
            seq_len_list = yb
            #test_batch.to(device)
            #seq_len_list.to(device)
            test_batch = test_batch.cuda(gpu_num)
            seq_len_list = seq_len_list.cuda(gpu_num)
            

            output = model(test_batch,seq_len_list)
            pred = output.argmax(dim=1, keepdim=True)
            pred_history.append(np.array(pred.cpu()))
            #print(np.array(pred.cpu()))

            batch+=1
            width = 40
            progress = (batch/int(len_data/len(xb)))
            xlen = int(progress*width)
            dlen = width - xlen
            print(f"[{'=' * xlen}{'.' * dlen}] {progress * 100:.1f}%",end='\r', flush=True)
        #print(pred_history)
        pred_history = np.array(pred_history).flatten()
        #print(pred_history)
        print("\n")
    
    
    return pred_history


#########################################################
#                         plot                          #
#########################################################




def plot_loss(loss_hist, metric_hist,plot_path):

    num_epochs= len(loss_hist["train"])

    fig = plt.figure()
    plt.subplot(1,1,1)
    plt.title("Train-Val Loss")
    plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
    plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()

    plt.subplot(1,1,2)
    plt.title("Train-Val Accuracy")
    plt.plot(range(1,num_epochs+1), metric_hist["train"],label="train")
    plt.plot(range(1,num_epochs+1), metric_hist["val"],label="val")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()

    plt.savefig(plot_path)


