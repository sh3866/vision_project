import argparse
import numpy as np
from tqdm import tqdm
from utils.utils import make_data_loader, valid_data_loader
# from model import BaseModel
import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils.tensorboard import SummaryWriter

from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

writer = SummaryWriter()


def acc(pred, label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()

def train(args, data_loader, valid_loader, model):
    
    # 손실함수 및 최적화함수
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 모델 GPU 사용 설정
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    args.device = device
    model = model.to(args.device)

    for epoch in range(args.epochs):
        train_losses = [] 
        train_acc = 0.0
        total = 0
        print(f"[Epoch {epoch+1} / {args.epochs}]")
        
        model.train()
        pbar = tqdm(data_loader)
        for i, (x, y) in enumerate(pbar):
            image = x.to(args.device)
            label = y.to(args.device)
            optimizer.zero_grad()

            output = model(image)
            
            label = label.squeeze()
            loss = criterion(output, label)
            
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            total += label.size(0)
            
            train_acc += acc(output, label)

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc / total
        
        # tensorboard 추가
        writer.add_scalar("Loss/train", epoch_train_loss, epoch)
        writer.add_scalar("Acc/train", epoch_train_acc, epoch)
        
        # validset 출력
        pbar2 = tqdm(valid_loader)
        with torch.no_grad():
            val_loss = 0.0
            val_acc = 0.0
            total_val_samples = 0
            
            model.eval()  # 모델을 평가 모드로 설정
            
            for j, (a, b) in enumerate(pbar2):
                images = a.to(args.device)
                labels = b.to(args.device)

                output = model(images)

                _, preds = output.max(dim=1)
                val_acc += preds.eq(labels).sum().item()
                val_loss += criterion(output, labels).item() * images.size(0)
                
                total_val_samples += labels.size(0)

        # 에폭당 검증 손실과 정확도 계산
        epoch_val_loss = val_loss / total_val_samples
        epoch_val_acc = val_acc / total_val_samples

        # TensorBoard에 기록
        writer.add_scalar("Loss/valid", epoch_val_loss, epoch)
        writer.add_scalar("Acc/valid", epoch_val_acc, epoch)

        # 결과 출력
        print(f'Epoch {epoch+1}')
        print(f'train_loss : {epoch_train_loss}')
        print(f'train_accuracy : {epoch_train_acc * 100:.2f}%')
        
        print(f'val_loss : {epoch_val_loss}')
        print(f'val_accuracy : {epoch_val_acc * 100:.2f}%')

        
        writer.flush()
        
        torch.save(model.state_dict(), f'{args.save_path}/model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2024 DSLAB project')
    parser.add_argument('--save-path', default='checkpoints/', help="Model's state_dict")
    parser.add_argument('--data', default='data/', type=str, help='data folder')
    args = parser.parse_args()

    # hyperparameters
    args.epochs = 3
    args.learning_rate = 0.0001
    args.batch_size = 16

    print("==============================")
    print("Save path:", args.save_path)
    print('Number of usable GPUs:', torch.cuda.device_count())
    print("Batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("Epochs:", args.epochs)
    print("==============================")
    
    # Make Data loader and Model
    train_loader, _ = make_data_loader(args)
    valid_loader = valid_data_loader(args)

    # custom model
    # model = BaseModel()
    
    # torchvision model
    
    # Resnet34
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # Mobilenet_v3_large
    # model = mobilenet_v3_large(weights = MobileNet_V3_Large_Weights.DEFAULT)
    # model.fc = nn.Linear(model.classifier[-1].in_features, 2)
    
    # Training The Model
    train(args, train_loader, valid_loader, model)
