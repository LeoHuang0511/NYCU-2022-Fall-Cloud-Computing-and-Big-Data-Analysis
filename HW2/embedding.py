import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import csv
import argparse


from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

#from model import ResNet3d18
from metric import KNN
from dataset import TrainDataset, TestDataset, ValDataset, UnlabeledDataset
from transforms import TrainAugment, TestAugment
from model import ResNetSimCLR
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default="./weights/best_weights/bestweights.pth", help="weights path")
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--model_type', type=str, default="resnet50")
    parser.add_argument('--embedding_path', type=str, default="./embedding.npy")
    args = parser.parse_args()


    weights_path = args.weights_path
    weights_name = os.path.split(os.path.split(os.path.split(weights_path)[0])[0])[1].strip('.pth')
    
    device = torch.device("cuda:"+args.device)


    ###Load Unlabeled data###
    num_workers = min(torch.get_num_threads(), 16)

    unlabeled_dataset = UnlabeledDataset(transform=TestAugment(size=96))

    
   
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False)

    ###Load model and Weights###
    print("using model type: ",args.model_type)
    model = ResNetSimCLR(base_model=args.model_type).to(device)

    print("loading weights name: ",weights_name)
    model.load_state_dict(torch.load(weights_path,map_location='cuda:'+args.device))


    ###Embedding the unlabeled data###
    embedding = torch.tensor([],requires_grad=False).to(device)
    len_unlabeled_data=len(unlabeled_loader)
    model.eval()
    with torch.no_grad():
            with tqdm(total=len_unlabeled_data, ncols=100, position=0, leave=True, desc="Embedding: ") as pbar:
                for x in unlabeled_loader:
                    x = x.to(device)
                    h = model(x,return_embedding=True)
                    embedding = torch.cat([embedding,h])
                    pbar.update(1)

    ###save the embedding###
    embedding_path = args.embedding_path
    embedding = embedding.cpu().detach().numpy()
    print("embedding size = ",embedding.shape)
    print("embedding dtype = ", embedding.dtype)
    np.save(embedding_path,embedding)
    print("Sucessfully save embedding at "+embedding_path )

    ####reload embedding###
    print("Reloading the embedding")
    reloaded_emb = np.load(embedding_path)
    print("embedding size = ",reloaded_emb.shape)
    print("embedding dtype = ", reloaded_emb.dtype)
    print("DONE!!")

if __name__ == '__main__':
    main()
    


