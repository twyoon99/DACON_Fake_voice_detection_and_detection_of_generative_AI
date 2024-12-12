import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
# import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, ResNet

class BiCNNLSTM(nn.Module):
    def __init__(self, input_dim=32000, hidden_dim=512, num_layers=1, dropout=0.5):
        super(BiCNNLSTM, self).__init__()

        # CNN layers with BatchNorm
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(512)

        self.maxpool = nn.MaxPool1d(kernel_size=2, st   ride=2)

        # LSTM layers
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2 * num_layers, 2)  # Adjusted for bidirectional LSTM

    def forward(self, x):
        # Ensure input shape: (batch_size, 1, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # CNN layers with BatchNorm and ReLU
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.maxpool(x)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.maxpool(x)
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.maxpool(x)
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.maxpool(x)

        
        # Reshape for LSTM
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, num_filters)
        
        # LSTM layer
        _, (h_n, _) = self.lstm(x)
        # h_n shape: (num_layers * num_directions, batch_size, hidden_dim)
        
        # Concatenate the hidden states of all layers
        h_n = h_n.permute(1, 0, 2).contiguous().view(x.size(0), -1)
        # h_n shape: (batch_size, num_layers * num_directions * hidden_dim)
        
        # Fully connected layer
        x = self.fc(h_n)
        
        # Sigmoid activation for binary classification
        out = torch.sigmoid(x)
        
        return out.squeeze()

class CNNLSTMDeep(nn.Module):
    def __init__(self, input_dim=32000, hidden_dim=512, num_layers=2, dropout=0.3):
        super(CNNLSTMWithBatchNorm, self).__init__()

        # CNN layers with BatchNorm
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm1d(512)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM layers
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * num_layers, 2)

    def forward(self, x):
        # Ensure input shape: (batch_size, 1, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # CNN layers with BatchNorm and LeakyReLU
        x = self.bn1(F.leaky_relu(self.conv1(x), negative_slope=0.01))
        x = self.maxpool(x)
        x = self.bn2(F.leaky_relu(self.conv2(x), negative_slope=0.01))
        x = self.maxpool(x)
        x = self.bn3(F.leaky_relu(self.conv3(x), negative_slope=0.01))
        x = self.maxpool(x)
        x = self.bn4(F.leaky_relu(self.conv4(x), negative_slope=0.01))
        x = self.maxpool(x)
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, num_filters)
        
        # LSTM layer
        _, (h_n, _) = self.lstm(x)
        # h_n shape: (num_layers * num_directions, batch_size, hidden_dim)
        
        # Concatenate the hidden states of all layers
        h_n = h_n.permute(1, 0, 2).contiguous().view(x.size(0), -1)
        # h_n shape: (batch_size, num_layers * num_directions * hidden_dim)
        
        # Fully connected layer
        x = self.fc(h_n)
        
        # Sigmoid activation for binary classification
        out = torch.sigmoid(x)
        
        return out.squeeze()


#best 
class CNNLSTMWithBatchNorm(nn.Module):
    def __init__(self, input_dim=32000, hidden_dim=512, num_layers=1, dropout=0.5):
        super(CNNLSTMWithBatchNorm, self).__init__()

        # CNN layers with BatchNorm
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM layers
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * num_layers, 2)

    def forward(self, x):
        # Ensure input shape: (batch_size, 1, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # CNN layers with BatchNorm and ReLU
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.maxpool(x)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.maxpool(x)
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.maxpool(x)
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.maxpool(x)
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, num_filters)
        
        # LSTM layer
        _, (h_n, _) = self.lstm(x)
        # h_n shape: (num_layers * num_directions, batch_size, hidden_dim)
        
        # Concatenate the hidden states of all layers
        h_n = h_n.permute(1, 0, 2).contiguous().view(x.size(0), -1)
        # h_n shape: (batch_size, num_layers * num_directions * hidden_dim)
        
        # Fully connected layer
        x = self.fc(h_n)
        
        # Sigmoid activation for binary classification
        out = torch.sigmoid(x)
        # out = torch.sigmoid(out)
        
        return out.squeeze()

# class BasicBlock1D(nn.Module):
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(BasicBlock1D, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(out_channels)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out

# class ResNet1D(nn.Module):
#     def __init__(self, block, layers, num_classes=2):
#         super(ResNet1D, self).__init__()
#         self.in_channels = 64
#         self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#         self.sigmoid = nn.Sigmoid()

#     def _make_layer(self, block, out_channels, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.in_channels != out_channels * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv1d(self.in_channels, out_channels * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(out_channels * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride, downsample))
#         self.in_channels = out_channels * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.in_channels, out_channels))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         # x: (batch_size, sequence_length)
#         x = x.unsqueeze(1)  # 추가된 부분: (batch_size, 1, sequence_length)
        
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         x = self.sigmoid(x)

#         return x

# def resnet18_1d(num_classes=2):
#     return ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_classes)


# from torchsummary import summary
# model = CNNLSTMWithResnet(in_channels=1, out_channels=2).to(device)

# summary(model, input_size=(3, 360, 480))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Display the summary of each model
# print("CNNLSTMWithResnet Model Summary:")
# summary(CNNLSTMWithResnet.to(device), input_size=(1, 32000))

# print("\nMobileNet1D Model Summary:")
# summary(MobileNet1D, (1, 32000))

# print("\nSpeakerVerificationLSTM Model Summary:")
# summary(SpeakerVerificationLSTM, (1, 32000))

# print("\nDepthwiseSeparableConv Model Summary:")
# summary(DepthwiseSeparableConv, (1, 32000))

# print("\nResNet1D Model Summary:")
# summary(ResBlock_1D, (1, 32000))
