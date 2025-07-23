# fusion_classifier.py

import torch.nn as nn

class FusionClassifierWrapper(nn.Module):
    def __init__(self, fusion_classifier):
        super(FusionClassifierWrapper, self).__init__()
        self.fusion_classifier = fusion_classifier

    def forward(self, fusion_embedding):
        return self.fusion_classifier(fusion_embedding)
