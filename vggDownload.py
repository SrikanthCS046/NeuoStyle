import torch
import torchvision.models as models
from torchvision.models import VGG19_Weights

# Download pretrained VGG19 weights
vgg = models.vgg19(weights=VGG19_Weights.DEFAULT)

# Save weights
torch.save(vgg.state_dict(), "vgg_normalised.pth")

print("VGG19 weights downloaded and saved as vgg_normalised.pth")