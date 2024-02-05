import argparse
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils.utils import make_data_loader
# from model import BaseModel
from torchvision.models import resnet34
from torchvision.models import mobilenet_v3_large

from sklearn.metrics import classification_report

def test(args, data_loader, model):
    true = np.array([])
    pred = np.array([])

    model.eval()

    pbar = tqdm(data_loader)
    for i, (x, y) in enumerate(pbar):
        image = x.to(args.device)
        label = y.to(args.device)

        output = model(image)

        label = label.squeeze()
        output = output.argmax(dim=-1)
        output = output.detach().cpu().numpy()
        pred = np.append(pred, output, axis=0)

        label = label.detach().cpu().numpy()
        true = np.append(true, label, axis=0)
                
    return pred, true


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2023 DL Term Project')
    parser.add_argument('--model-path', default='checkpoints/model.pth', help="Model's state_dict")
    parser.add_argument('--data', default='data/', type=str, help='data folder')
    args = parser.parse_args()

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    # hyperparameters
    args.batch_size = 4
    
    # Make Data loader and Model
    _, test_loader = make_data_loader(args)

    # instantiate model

    model = resnet34()
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # model = mobilenet_v3_large()
    # model.fc = nn.Linear(model.classifier[-1].in_features, 2)
    
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    
    # Test The Model
    pred, true = test(args, test_loader, model)
        
    accuracy = (true == pred).sum() / len(pred)
    print("Test Accuracy : {:.5f}".format(accuracy))
    
    # classification report 
    print(classification_report(true, pred))

    # 정성 분석 데이터 가져오기 (나온 인덱스 + 2 해주면 됨)
    for i in range(len(true)):
        if true[i] != pred[i]:
            print(f"Index {i}: True = {true[i]}, Pred = {pred[i]}")