import torch
import torch.nn as nn


class QuickDrawCNN(nn.Module):
    """
    Compact Convolutional Neural Network for 28x28 grayscale sketch classification.
    Input: (B, 1, 28, 28), float32 in [0, 1]
    Output: (B, num_classes) raw logits
    """
    def __init__(self, num_classes=8):
        super(QuickDrawCNN, self).__init__()
        
        # Block 1: 1 -> 32 channels, 28x28 -> 14x14
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: 32 -> 64 channels, 14x14 -> 7x7
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: 64 -> 128 channels, 7x7 -> 3x3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = QuickDrawCNN(num_classes=8)
    dummy_input = torch.randn(4, 1, 28, 28)
    out = model(dummy_input)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"QuickDrawCNN initialized successfully.")
    print(f"Output shape: {out.shape}")
    print(f"Total trainable parameters: {total_params:,}")
