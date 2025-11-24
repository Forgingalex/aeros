"""CNN model for heading estimation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HeadingCNN(nn.Module):
    """Lightweight CNN for heading angle regression.
    
    Architecture: MobileNet-style depthwise separable convolutions
    with a regression head predicting heading angle in radians.
    """
    
    def __init__(self, input_channels: int = 3, output_dim: int = 1):
        """Initialize model.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            output_dim: Output dimension (1 for heading angle)
        """
        super(HeadingCNN, self).__init__()
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Depthwise separable convolutions
        self.conv2 = self._make_separable_conv(32, 64, stride=2)
        self.conv3 = self._make_separable_conv(64, 128, stride=2)
        self.conv4 = self._make_separable_conv(128, 256, stride=2)
        self.conv5 = self._make_separable_conv(256, 512, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Regression head
        self.fc1 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, output_dim)
        
    def _make_separable_conv(
        self, in_channels: int, out_channels: int, stride: int = 1
    ) -> nn.Sequential:
        """Create depthwise separable convolution block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            stride: Stride for convolution
            
        Returns:
            Sequential block with depthwise and pointwise convs
        """
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=stride,
                padding=1, groups=in_channels, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Heading angle prediction (B, 1)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

