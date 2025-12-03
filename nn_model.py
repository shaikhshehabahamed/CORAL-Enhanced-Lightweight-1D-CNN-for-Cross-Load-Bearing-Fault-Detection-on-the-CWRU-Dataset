# nn_model.py (updated)
import torch
from torch import nn

# import grad_reverse from the separate module (avoid duplicate definitions)
from grad_reverse import grad_reverse

# 2-layer backbone
class CNN_1D_2L(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.n_in = n_in

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.MaxPool1d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.AvgPool1d(2)
        )

        # flattened feature dimension after /4
        self.flat_dim = (self.n_in // 4) * 128

        # classifier heads: GAP head and flattened head
        self.classifier_gap = nn.Linear(128, 4)
        self.classifier_flat = nn.Linear(self.flat_dim, 4)

    def forward(self, x, return_feats=False, use_gap=True):
        x = x.view(-1, 1, self.n_in)
        x = self.layer1(x)
        x = self.layer2(x)

        # features for CORAL / classifier input
        if use_gap:
            feats = x.mean(dim=2)              # (B, 128)
            logits = self.classifier_gap(feats)
        else:
            feats = x.view(x.size(0), -1)      # flattened (B, flat_dim)
            logits = self.classifier_flat(feats)

        if return_feats:
            return logits, feats
        return logits


# 3-layer backbone
class CNN_1D_3L(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.n_in = n_in

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.MaxPool1d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.MaxPool1d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.MaxPool1d(2)
        )

        # flattened feature dimension after /8
        self.flat_dim = (self.n_in // 8) * 128

        # classifier heads: GAP head and flattened head
        self.classifier_gap = nn.Linear(128, 4)
        self.classifier_flat = nn.Linear(self.flat_dim, 4)

    def forward(self, x, return_feats=False, use_gap=True):
        x = x.view(-1, 1, self.n_in)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # features for CORAL / classifier input
        if use_gap:
            feats = x.mean(dim=2)              # (B, 128)
            logits = self.classifier_gap(feats)
        else:
            feats = x.view(x.size(0), -1)      # flattened (B, flat_dim)
            logits = self.classifier_flat(feats)

        if return_feats:
            return logits, feats
        return logits


# DANN model
class CNN_DANN(nn.Module):
    def __init__(self, n_in: int, use_gap: bool = True):
        super().__init__()
        # store use_gap so forward can pass it to the feature extractor
        self.use_gap = use_gap
        self.feature_extractor = CNN_1D_3L(n_in)

        # choose feature dim according to use_gap
        feat_dim = 128 if use_gap else self.feature_extractor.flat_dim

        # domain classifier (2 domains: source/target)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 2)
        )

    def forward(self, x, lambd=0.0, use_gap=None):
        # allow caller to override use_gap for this forward call
        if use_gap is None:
            use_gap = self.use_gap

        # obtain logits + features from backbone
        # feature_extractor returns (logits, feats) when return_feats=True
        logits, feats = self.feature_extractor(x, return_feats=True, use_gap=use_gap)

        # GRL for domain classifier
        rev_feats = grad_reverse(feats, lambd)
        dom_logits = self.domain_classifier(rev_feats)

        return logits, dom_logits, feats