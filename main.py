from dataset import *
from model import *
from utils import *
from torchsummary import summary
from train import *
from augment import *
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR, StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torchvision import transforms
from inference import *
import os
import numpy as np
import random 
from torchvision import models


import warnings
warnings.filterwarnings(action='ignore') 


# 시드 고정 함수
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)  # Seed=42 고정

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로드 및 전처리
df = pd.read_csv('/home/vaill/cowork/mai/project/project/train.csv')
train_df, val_df, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=42)
test_df = pd.read_csv('/home/vaill/cowork/mai/project/project/test.csv')
test_clean_df = pd.read_csv('/home/vaill/cowork/mai/project/project/test_cleaned.csv')

# 메인 함수
def main(args):
    print("Start Code")
    print("My model")

    if args.data_type == "1d":
        if args.model == 'CNNLSTMDeep':
            model = CNNLSTMDeep()
        elif args.model == 'CNNLSTMWithBatchNorm':
            model = CNNLSTMWithBatchNorm()
        elif args.model == 'resnet':
            model = resnet18_1d(num_classes=2)
        elif args.model == 'BiCNNLSTM':
            model = BiCNNLSTM()


        N_CLASSES = 2
        real_file_paths_train, fake_file_paths_train = get_file_paths_and_labels(train_df, N_CLASSES)
        real_file_paths_val, fake_file_paths_val = get_file_paths_and_labels(val_df, N_CLASSES)


        # data_augmentation = RandomCrop(crop_size=8000) 
        data_augmentation = DataAugmentation_V2()
        data_augmentation_test = DataAugmentation_test()


        # 학습 데이터셋을 생성합니다.
        train_dataset = CustomDataset(real_file_paths_train, fake_file_paths_train, transform=data_augmentation)
        val_dataset = CustomDataset(real_file_paths_val, fake_file_paths_val, transform=data_augmentation)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.BATCH_SIZE,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.BATCH_SIZE,
            shuffle=False
        )
        test_clean_dataset = CustomDatatest(test_clean_df['path'].tolist(), transform=data_augmentation_test)
        # test_dataset = CustomDatatest(test_df['path'].tolist(), transform=data_augmentation_test) # CustomDatatest
        test_loader = DataLoader(
            test_clean_dataset,
            batch_size=256,
            shuffle=False
        )

        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.LR, momentum=0.9)

        if args.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
        elif args.scheduler == 'StepLR':
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        elif args.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1, verbose='deprecated')
        
        
    if args.phase == 'train':
        infer_model = train_model(model, optimizer, scheduler, train_loader, val_loader, epoch=args.N_EPOCHS, device=device, patience=5)
        torch.save(infer_model.state_dict(), args.model_path)
    elif args.phase == 'inference':
        model.load_state_dict(torch.load(args.model_path))
        preds = inference(model, test_loader, device)
        print(len(preds))
        submit = pd.read_csv('/home/vaill/cowork/mai/project/project/sample_submission.csv')
        #preds2 = [item[0] for item in preds]
        submit.iloc[:, 1:] = preds
        submit.to_csv('submission.csv', index=False)  # 제출 파일 저장
        print(submit.head(20))
        submit.to_csv(args.submit_path, index=False)

# 커맨드라인 인자 처리 및 메인 함수 호출
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 1D or 2D CNN for Audio Classification.")
    parser.add_argument("--data_type", type=str, choices=["1d", "2d"], help="Type of the data: '1d' for raw audio, '2d' for spectrogram")
    parser.add_argument('--scheduler', type=str, help="Type of scheduler")
    parser.add_argument('--model', type=str, help="Load pre-trained model")
    parser.add_argument("--optimizer", type=str, help="optimizer")
    parser.add_argument('--n_class', type=int, help="Number of class")
    parser.add_argument('--BATCH_SIZE', type=int, help="Number of BATCH_SIZE")
    parser.add_argument('--LR', type=float, help="Learning rate")
    parser.add_argument('--SR', type=int, help="Sample rate")
    parser.add_argument('--N_EPOCHS', type=int, help="Number of epochs")
    parser.add_argument('--phase', type=str, choices=['train', 'inference'], help="Phase ; train or evaluate")
    parser.add_argument('--model_path', type=str, help="Trained model's path")
    parser.add_argument("--submit_path", type=str, default="submission.csv", help="Path to save the submission CSV file during inference phase.")
    args = parser.parse_args()
    main(args)
