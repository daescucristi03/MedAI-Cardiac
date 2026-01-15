import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out

class CNNLSTM(nn.Module):
    def __init__(self, input_channels=12, num_classes=1):
        super(CNNLSTM, self).__init__()
        
        # Enhanced CNN Encoder (ResNet-style)
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        
        # LSTM for temporal dependencies
        # Input features: 256 (from CNN)
        # Reverted dropout to 0.5 for better confidence
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
        
        # Attention Mechanism
        self.attention = nn.Linear(128 * 2, 1)

        # Classifier
        # Bidirectional LSTM outputs 2 * hidden_size
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5), # Reverted dropout
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, 12, 5000)
        
        # CNN Feature Extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # Output shape: (batch, 256, reduced_time)
        
        # Prepare for LSTM: (batch, time, features)
        x = x.permute(0, 2, 1)
        
        # LSTM
        # out: (batch, seq_len, num_directions * hidden_size)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        
        # Attention Mechanism
        attn_weights = F.softmax(self.attention(out), dim=1)
        out = torch.sum(attn_weights * out, dim=1)
        
        # Classification
        out = self.fc(out)
        
        return out
