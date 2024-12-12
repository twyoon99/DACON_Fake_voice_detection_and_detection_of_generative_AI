import librosa
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from IPython.display import Audio, display
import random
import torchaudio.transforms as T
from sklearn.preprocessing import MinMaxScaler


# 데이터셋을 로드합니다
df = pd.read_csv('/home/vaill/cowork/mai/project/project/train_10000.csv')
train_df, val_df, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=42)
test_df = pd.read_csv('/home/vaill/cowork/mai/project/project/test.csv')
test_clean_df = pd.read_csv('/home/vaill/cowork/mai/project/project/test_cleaned.csv')

def get_file_paths_and_labels(df, N_CLASSES):
    real_file_paths = []
    fake_file_paths = []
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        img_path = row['path']
        label = row['label']
        
        if label == 'real':
            real_file_paths.append(img_path)
        else:
            fake_file_paths.append(img_path)

    return real_file_paths, fake_file_paths

# 데이터 어그멘테이션 클래스를 정의합니다.

class DataAugmentation_V2:
    def __init__(self, noise_factor=0.02, mask_param=30):
        self.noise_factor = noise_factor
        self.mask_param = mask_param
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=self.mask_param)
    
    def add_noise(self, sample):
        if isinstance(sample, torch.Tensor):
            sample = sample.numpy()
        
        noise_amp = self.noise_factor * np.random.uniform() * np.amax(sample)
        sample = sample + noise_amp * np.random.normal(size=sample.shape)
        
        return torch.tensor(sample, dtype=torch.float32)
    

    def time_mask(self, waveform):
        return self.time_masking(waveform)

    def __call__(self, sample, sampling_rate):
        if random.random() < 0.2:
            sample = self.add_noise(sample)
        if random.random() < 0.2:
            sample = self.time_mask(sample)
        return torch.tensor(sample.flatten(), dtype=torch.float32)
    



class RandomCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, waveform):
        max_offset = waveform.size(1) - self.crop_size
        offset = random.randint(0, max_offset)
        return waveform[:, offset:offset + self.crop_size]


class CustomDataset(Dataset):
    def __init__(self, real_file_paths, fake_file_paths, transform=None):
        self.real_file_paths = real_file_paths
        self.fake_file_paths = fake_file_paths
        self.transform = transform

    def __len__(self):
        return len(self.real_file_paths) + len(self.fake_file_paths)

    def __getitem__(self, index):
        
        scenario = np.random.choice([1, 2, 3, 4, 5, 6, 7])

        if scenario == 1:  # 1명의 진짜 목소리
            real_index = index % len(self.real_file_paths)
            real_path = self.real_file_paths[real_index]
            waveform, sr = torchaudio.load(real_path)
            label = [0, 1]
        
        elif scenario == 2:  # 1명의 가짜 목소리
            fake_index = index % len(self.fake_file_paths)
            fake_path = self.fake_file_paths[fake_index]
            waveform, sr = torchaudio.load(fake_path)
            label = [1, 0]
        
        elif scenario == 3:  # 1명의 진짜 목소리와 1명의 가짜 목소리
            real_index = index % len(self.real_file_paths)
            fake_index = index % len(self.fake_file_paths)
            real_path = self.real_file_paths[real_index]
            fake_path = self.fake_file_paths[fake_index]
            waveform_real, sr_real = torchaudio.load(real_path)
            waveform_fake, sr_fake = torchaudio.load(fake_path)
            if sr_real != sr_fake:
                resampler = torchaudio.transforms.Resample(orig_freq=sr_fake, new_freq=sr_real)
                waveform_fake = resampler(waveform_fake)
                raise ValueError("Sample rates do not match.")
            
            min_length = min(waveform_real.size(1), waveform_fake.size(1))
            waveform_real = waveform_real[:, :min_length]
            waveform_fake = waveform_fake[:, :min_length]
            waveform = waveform_real + waveform_fake
            waveform = waveform / torch.max(torch.abs(waveform))
            sr = sr_real
            label = [1, 1]
        
        elif scenario == 4:  # 2명의 진짜 목소리
            real_index1 = index % len(self.real_file_paths)
            real_index2 = (index + 1) % len(self.real_file_paths)
            real_path1 = self.real_file_paths[real_index1]
            real_path2 = self.real_file_paths[real_index2]
            waveform_real1, sr1 = torchaudio.load(real_path1)
            waveform_real2, sr2 = torchaudio.load(real_path2)
            if sr1 != sr2:
                resampler = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=sr1)
                waveform_real2 = resampler(waveform_real2)
                raise ValueError("Sample rates do not match.")
            
            min_length = min(waveform_real1.size(1), waveform_real2.size(1))
            waveform_real1 = waveform_real1[:, :min_length]
            waveform_real2 = waveform_real2[:, :min_length]
            waveform = waveform_real1 + waveform_real2
            waveform = waveform / torch.max(torch.abs(waveform))
            sr = sr1
            label = [0, 1]
        
        elif scenario == 5:  # 2명의 가짜 목소리
            fake_index1 = index % len(self.fake_file_paths)
            fake_index2 = (index + 1) % len(self.fake_file_paths)
            fake_path1 = self.fake_file_paths[fake_index1]
            fake_path2 = self.fake_file_paths[fake_index2]
            waveform_fake1, sr1 = torchaudio.load(fake_path1)
            waveform_fake2, sr2 = torchaudio.load(fake_path2)
            if sr1 != sr2:
                resampler = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=sr1)
                waveform_fake2 = resampler(waveform_fake2)
                raise ValueError("Sample rates do not match.")
            
            min_length = min(waveform_fake1.size(1), waveform_fake2.size(1))
            waveform_fake1 = waveform_fake1[:, :min_length]
            waveform_fake2 = waveform_fake2[:, :min_length]
            waveform = waveform_fake1 + waveform_fake2
            waveform = waveform / torch.max(torch.abs(waveform))
            sr = sr1
            label = [1, 0]
        
        elif scenario ==6 :  # no voice and add noise
            waveform = torch.zeros(1, 32000)
            #add noise
            waveform = waveform + (0.1**0.5)*torch.randn(1,32000) 
            sr = 32000
            label = [0, 0]
            
        elif scenario == 7 : #no voice and no noise
            waveform = torch.zeros(1, 32000)
            sr = 32000
            label = [0, 0]
            

        desired_length = 32000
        if waveform.size(1) > desired_length:
            max_start = waveform.size(1) - desired_length
            start = torch.randint(0, max_start, (1,)).item()
            waveform = waveform[:, start:start + desired_length]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, desired_length - waveform.size(1)))

        if self.transform:
            waveform = self.transform(waveform, sr)

        return waveform, torch.tensor(label, dtype=torch.float32)



class DataAugmentation_test:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def __call__(self, sample, sampling_rate):
        return torch.tensor(sample.flatten(), dtype=torch.float32).unsqueeze(0)  # Add channel dimension

class CustomDatatest(Dataset):
    def __init__(self, file_paths, transform=None, desired_length=32000, segment_length=16000, overlap=0.5):
        self.file_paths = file_paths
        self.transform = transform
        self.desired_length = desired_length
        self.segment_length = segment_length
        self.overlap = overlap

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        waveform, sr = torchaudio.load(self.file_paths[index])
        
        segments = []
        num_segments = int((waveform.size(1) - self.segment_length) / (self.segment_length * self.overlap)) + 1
        for i in range(num_segments):
            start = int(i * self.segment_length * self.overlap)
            end = start + self.segment_length
            segment = waveform[:, start:end]
            
            if self.transform:
                segment = self.transform(segment, sr)
            else:
                segment = segment.unsqueeze(0)  # Add channel dimension if transform is not applied
            
            segments.append(segment)
        
        return segments
