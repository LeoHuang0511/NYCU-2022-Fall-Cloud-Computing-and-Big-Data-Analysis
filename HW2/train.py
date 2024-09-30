import argparse
import os

import torch
import torchvision.transforms as transforms
import copy

from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lossFn import xt_xent
from metric import KNN
from dataset import TrainDataset, TestDataset, ValDataset
from transforms import TrainAugment, TestAugment
from model import ResNetSimCLR
from torch.optim.lr_scheduler import  ReduceLROnPlateau, CosineAnnealingLR







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--weights_path', type=str, default="./weights/")
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--model_type', type=str, default="resnet50")
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--extra_tag', type=str, default='')
    args = parser.parse_args()

    
    device = torch.device('cuda:'+args.device)

    tag = args.extra_tag+'_'+args.model_type+'_'+f'lr{args.lr}_batch{args.batch_size}'


    os.makedirs(args.weights_path, exist_ok=True)
    ckpoint_path = args.weights_path+'/checkpoint/'
    best_weights_path = args.weights_path+'/best_weights/'
    os.makedirs(ckpoint_path)
    os.makedirs(best_weights_path)

    os.makedirs(args.logdir,exist_ok=True)
    train_writer = SummaryWriter(os.path.join(args.logdir))
    
    num_workers = min(torch.get_num_threads(), 16)

    
    train_dataset = TrainDataset(transform=TrainAugment(size=96))
    val_dataset = ValDataset(transform=TrainAugment(size=96))
    test_dataset = TestDataset(transform=TestAugment(size=96))

    indices = torch.randperm(
        len(train_dataset), generator=torch.Generator().manual_seed(42)).tolist()

    # train subset
    train_dataset = Subset(dataset=train_dataset, indices=indices)
    #train_indices = indices[:int(len(dataset) * 0.8)]
   
    train_loader = DataLoader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True)

    val_loader = DataLoader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False)
    
    test_loader = DataLoader(
        test_dataset,
        len(test_dataset),
        shuffle=False,
        num_workers=num_workers,
        drop_last=False)

    model = ResNetSimCLR(base_model=args.model_type).to(device)
    #print(model)
    loss_fn = xt_xent()
    
    optimizer = Adam(model.parameters(), args.lr)    


    #train_step = 1
    #test_step = 1
    best_acc = 0.0
    current_lr = args.lr
    print(f"Using {args.model_type} training on cuda:{args.device} with lr={args.lr}, batch size={args.batch_size}")
    for epoch in range(0, args.epochs + 1):
        
        #Training
        #print("Training......\n")
        print(f"Epoch {epoch:2d}/{args.epochs:2d}")
        model.train()
        running_loss=0.0
        len_train_data = len(train_loader)
        with tqdm(total=len_train_data ,ncols=100, position=0, leave=True, desc="Training: ") as pbar:
            #pbar.set_description(f"Epoch {epoch:2d}/{args.epochs:2d}")
            for x1, x2 in train_loader:
                x1, x2 = x1.to(device), x2.to(device)
                z1, z2 = model(x1), model(x2)
                loss_b = loss_fn(z1, z2,temperature=args.temp)
                optimizer.zero_grad()
                loss_b.backward()
                optimizer.step()
                running_loss+=loss_b
                #train_step += 1
                pbar.set_postfix_str(f"loss_b: {loss_b:.4f}")
                pbar.update(1)
            
            loss=running_loss/float(len_train_data)
            
            
            current_lr=optimizer.param_groups[0]['lr']
            

            #lr_scheduler.step(loss)
            
            
            #warmup_scheduler.step(metrics=loss)
            
            train_writer.add_scalars(main_tag='All/train_loss',
                                    tag_scalar_dict={tag+'_train_loss': float(loss)},
                                    global_step=epoch)
            
        #best_loss=float('inf') 
        
        #Validation(Testing)
        running_val_loss=0.0
        running_acc=0.0
        len_val_data=len(val_loader)
        len_test_data=len(test_loader)
        model.eval()
        with torch.no_grad():
            #print("Testing......\n")
            with tqdm(total=len_val_data, ncols=100, position=0, leave=True, desc="Validating: ") as pbar:
                for x1, x2 in val_loader:
                    x1, x2= x1.to(device), x2.to(device)
                    h1, h2 = model(x1,return_embedding=False), model(x2,return_embedding=False)
                    loss_b = loss_fn(h1, h2,temperature=args.temp)
                    running_val_loss+=loss_b
                    pbar.set_postfix_str(f"loss_b: {loss_b:.4f}")
                    pbar.update(1)
                val_loss=running_val_loss/float(len_val_data)

                

                train_writer.add_scalars(main_tag='All/val_loss',
                                        tag_scalar_dict={tag+'_val_loss': float(val_loss)},
                                        global_step=epoch)


            with tqdm(total=len_test_data, ncols=100, position=0, leave=True, desc="Testing: ") as pbar:
                for x, y in test_loader:
                    x, y= x.to(device), y.to(device)
                    h = model(x,return_embedding=True)
                    #print(f"shape of embeddings{h.shape}, shape of labels:{y.shape}")
                    acc_b = KNN(emb=h, cls=y, batch_size=args.batch_size)
                    running_acc+=acc_b
                    pbar.set_postfix_str(f"acc_b: {acc_b:.4f}")
                    pbar.update(1)
                
                acc = running_acc/float(len_test_data)
                train_writer.add_scalars(main_tag='All/acc',
                                        tag_scalar_dict={tag+'_acc': float(acc)},
                                        global_step=epoch)
                if acc >= best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), best_weights_path+'bestweights.pth')
                #true_y.append(y.cpu())
                #pred_y.append(y_hat.detach().argmax(dim=-1).cpu())
                
                #train_writer.add_scalar("loss", loss, step)
                #test_step += 1
        train_writer.add_scalars(main_tag=tag+'/loss ', 
                                tag_scalar_dict={'train_loss': float(loss),
                                                'val_loss': float(val_loss)},
                                global_step=epoch)
        train_writer.add_scalars(main_tag=tag+'/acc', 
                                tag_scalar_dict={'acc': float(acc)},
                                global_step=epoch)
        train_writer.add_scalars(main_tag=tag+'/lr', 
                                tag_scalar_dict={'lr' : float(current_lr)},
                                global_step=epoch)
        
        print(", ".join([
            f"Epoch {epoch:3d}/{args.epochs:3d}",
            f"train_loss: {loss:.4f}",
            f"val_loss: {val_loss:.4f}",
            f"test_acc: {acc:.4f}",
            f"lr: {current_lr}"
        ]))
        #true_y = torch.cat(true_y, dim=0)
        #pred_y = torch.cat(pred_y, dim=0)
        #train_acc = (true_y == pred_y).float().mean()
        #train_writer.add_scalar("acc", train_acc, epoch)
        if epoch%5==0:
            torch.save(model.state_dict(), ckpoint_path+str(epoch)+'.pth')
        
        

                    
        

            






if __name__ == '__main__':
    main()
