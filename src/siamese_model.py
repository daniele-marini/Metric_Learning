import torch
from torch.nn import Module, Sequential, Linear, ReLU
from torchsummary import summary
from torchvision import models

class SiameseNetwork(Module):

    def __init__(self, output : int, backbone: models,freezed=False):
        super(SiameseNetwork, self).__init__()

        self.backbone = backbone

        if freezed:
          self.set_requires_grad_for_layer(backbone.conv1, False)
          self.set_requires_grad_for_layer(backbone.bn1, False)
          self.set_requires_grad_for_layer(backbone.layer1, False)
          self.set_requires_grad_for_layer(backbone.layer2, False)
          self.set_requires_grad_for_layer(backbone.layer3, False)
          self.set_requires_grad_for_layer(backbone.layer4, False)

        self.backbone.fc = Sequential (
            Linear(self.backbone.fc.in_features, 512),
            ReLU(inplace = True),
            Linear(512, 256),
            ReLU(inplace = True),
            Linear(256, output)
        )

    def backbone_summary(self):
      return summary(self.backbone , input_size=(3, 224, 224))


    def set_requires_grad_for_layer(self, layer: torch.nn.Module, train: bool) -> None:
      for p in layer.parameters():
          p.requires_grad = train

    def forward_once(self, x : torch.Tensor):
        return self.backbone(x)


    def forward(self, inputs : torch.Tensor):

        if len(inputs) == 3:
            return torch.stack((self.forward_once(inputs[0]),
                                self.forward_once(inputs[1]),
                                self.forward_once(inputs[2])))

        raise ValueError(f'This is the SiameseBaseline, the number of input tensor 3, you gave {inputs.size()} instead')