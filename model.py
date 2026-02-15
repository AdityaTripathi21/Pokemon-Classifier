import torch
import torch.nn as nn

    # CNN for multi-label image classification
class SimpleCNN(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()

        # input - [B, C, H, W]
        # B - Batch size, C - Number of channels, H - Image height, W - Image width
        # In our case, [B, 3, 128, 128]

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),    # in channels represent 3 color channels
            # out_channels - number of feature maps/kernels used, these are also the weights and every feature map has a bias
            # kernel_size - pretty intuitive, 3x3 in this case
            # padding - extra layer of 0s around input image to make kernel fit

            nn.ReLU(),  # non-linear layer applied element wise to all feature maps

            # input tensor goes from [B, C, H, W] to [B, out_channels, H, W], so [B, 16, 128, 128]

            nn.MaxPool2d(kernel_size=2),    # pooling to downscale parameters and make it less computationally expensive
            # kernel_size - size of window to take max over, 2x2 in this case

            # max pool halves dimension size, so [B, out_channels, H/2, W/2] - > [B, 16, 64, 64]

            # out_channels -> 16 - 32
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # out_channels -> 32 - 64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Output tensor -> [B, 64, 16, 16]
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1)) # average out a feature map to one number, 
        # so [B, 64, 16, 16] -> [B, 64, 1, 1]

        self.classifier = nn.Sequential(
            nn.Flatten(),   # flatten matrices into a linear layer, [B, 64, 1, 1] -> [B, 64]
            nn.Linear(64, 128), # [B, 64] -> [B, 128]
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes) # [B, 128] -> [B, num_classes], remember logits not probabilities
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x



    # Use BCELoss because we need binary decisions, even though it's multi-label
    # 18 yes/no questions get asked simultaneously 
    # Afterwards, sigmoid is used to convert logits into probabilities
    # Sigmoid is used because multiple classes can also be true

    # The CE Loss formula comes from the KL divergence, which is a natural measure of distance between distributions,
    # in our case, we want to measure the distance between the true class given our input between the predicted class given the input
    # But to do this, you need to apply a sigmoid function to convert logits into probabilities.




