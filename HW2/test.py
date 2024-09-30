import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import csv
import argparse


from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from metric import KNN
from dataset import TrainDataset, TestDataset, ValDataset
from transforms import TrainAugment, TestAugment
from model import ResNetSimCLR



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default="./weights/best_weights/bestweights.pth", help="weights path")
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--model_type', type=str, default="resnet50")
    args = parser.parse_args()


    weights_path = args.weights_path
    weights_name = os.path.split(os.path.split(os.path.split(weights_path)[0])[0])[1].strip('.pth')
    
    device = torch.device("cuda:"+args.device)

    ####Load Test data###
    
    num_workers = min(torch.get_num_threads(), 16)

    test_dataset = TestDataset(transform=TestAugment(size=96))

    
    test_loader = DataLoader(
        test_dataset,
        len(test_dataset),
        shuffle=False,
        num_workers=num_workers,
        drop_last=False)

    ###Load model and Weights###
    print("using model type: ",args.model_type)
    model = ResNetSimCLR(base_model=args.model_type).to(device)

    print("loading weights name: ",weights_name)
    model.load_state_dict(torch.load(weights_path,map_location='cuda:'+args.device))

    ###Computing the accuracy###
    len_test_data=len(test_loader)
    model.eval()
    with torch.no_grad():
       with tqdm(total=len_test_data, ncols=100, position=0, leave=True, desc="Testing: ") as pbar:
                for x, y in test_loader:
                    x, y= x.to(device), y.to(device)
                    h = model(x,return_embedding=True)
                    #print(f"shape of embeddings{h.shape}, shape of labels:{y.shape}")
                    acc_b = KNN(emb=h, cls=y, batch_size=args.batch_size)
                    pbar.set_postfix_str(f"acc_b: {acc_b:.4f}")
                    #pbar.update(1)

    print(f"Test Accuracy using {weights_name}: ", acc_b)

if __name__ == '__main__':
    main()