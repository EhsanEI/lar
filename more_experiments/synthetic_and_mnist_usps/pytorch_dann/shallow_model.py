import torch.nn as nn
import torch.nn.functional as F
from dannutils import ReverseLayerF
import params

if params.target_domain == 'mnistm':
    input_channels = 3
else:
    input_channels = 1


class Extractor(nn.Module):
    def __init__(self, hidden_dim):
        super(Extractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Linear(in_features=1 * 28 * 28, out_features=hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.flatten(1)
        x = self.extractor(x)
        return x


class Classifier(nn.Module):
    def __init__(self, hidden_dim):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=2)
        )

    def forward(self, x):
        x = self.classifier(x)
        return F.softmax(x)


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=2)
        )

    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        x = self.discriminator(reversed_input)
        return F.softmax(x)
