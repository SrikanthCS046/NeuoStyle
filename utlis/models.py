
import torch
import torch.nn as nn


class VGGEncoder(nn.Module):
    """
    VGG encoder matching the original AdaIN architecture.

    Expected checkpoint:
        vgg_normalised.pth from the original AdaIN repository.
    """

    def __init__(self, vgg_path):
        super().__init__()

        # Original AdaIN VGG architecture.
        self.vgg = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1),
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(inplace=False),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )

        weights = torch.load(
            vgg_path,
            map_location="cpu",
            weights_only=True,
        )

        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]

        # Do not silently accept a mismatched VGG checkpoint.
        self.vgg.load_state_dict(weights, strict=True)

        # Freeze the encoder.
        for parameter in self.vgg.parameters():
            parameter.requires_grad = False

        self.enc_1 = nn.Sequential(*list(self.vgg.children())[:4])
        self.enc_2 = nn.Sequential(*list(self.vgg.children())[4:11])
        self.enc_3 = nn.Sequential(*list(self.vgg.children())[11:18])
        self.enc_4 = nn.Sequential(*list(self.vgg.children())[18:31])

    def forward(self, x, is_test=False):
        h1 = self.enc_1(x)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)

        if is_test:
            return h4

        return h1, h2, h3, h4


class Decoder(nn.Module):
    """
    Decoder architecture from the original AdaIN implementation.

    State-dict keys are named net.0, net.1, etc.
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3),
            nn.ReLU(inplace=False),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(inplace=False),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(inplace=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(inplace=False),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3),
        )

    def forward(self, x):
        return self.net(x)
