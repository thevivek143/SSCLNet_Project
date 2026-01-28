import torch.nn as nn

class SSCLClassifier(nn.Module):
    """
    Generic Classifier built on top of the encoder.
    """
    def __init__(self, encoder, num_classes):
        super(SSCLClassifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(512, num_classes) # ResNet18 output is 512

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        return output

def get_brain_classifier(encoder, num_classes=4):
    return SSCLClassifier(encoder, num_classes)

def get_knee_classifier(encoder, num_classes=5):
    return SSCLClassifier(encoder, num_classes)
