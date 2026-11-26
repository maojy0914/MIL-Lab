from src.models.mil_template import MIL
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x

class MCAB(nn.Module):
    def __init__(self,dim=512,temperature=3):
        super().__init__()

        self.local_feature3 = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=3,stride=1, padding=3 // 2, groups=dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, groups=1)
        )

        self.local_feature5 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=5 // 2, groups=dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, groups=1)
        )


        ## self.eca_layer = ECA_Layer(dim)
        # self.cam = CAM_1(n_channel=dim,mlp_r=2,temperature=temperature)
        self.cam = CAM(n_channel=dim, mlp_r=2, temperature=temperature)

    def forward(self,x):
        B, N, C = x.shape

        _H, _W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        add_length = _H * _W - N
        zero_tensor = torch.zeros(B, add_length, C,device=x.device)

        x_pad = torch.cat([x, zero_tensor], dim=1) # [B,_H*_W,C]

        cnn_feat = x_pad.transpose(1, 2).view(B, C, _H, _W)
        x3_3 = self.local_feature3(cnn_feat)
        x5_5 = self.local_feature5(cnn_feat)
        x = cnn_feat + x3_3 + x5_5

        y1,y2,y3 = self.cam(x) # [B,C,1,1]
        x = y1 * cnn_feat + y2 * x3_3 + y3 * x5_5
        x = x.flatten(2).transpose(1, 2)    # [B,_H*_W,C]
        x = x[:, :N]
        return x

class CAM(nn.Module):
    def __init__(self,n_channel,mlp_r=2,temperature=1):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(n_channel, n_channel // mlp_r),
            nn.ReLU(),
            nn.Linear(n_channel // mlp_r, 3 * n_channel)
        )
        self.temperature = temperature

    def forward(self,x):
        B,C,_,_ = x.shape
        max = self.max_pool(x)
        avg = self.avg_pool(x)
        max = self.linear(max.view(B,C)).view(B,3*C,1,1)
        avg = self.linear(avg.view(B,C)).view(B,3*C,1,1)
        y = torch.sigmoid((max + avg) * self.temperature)
        y_1 = y[:,:C]
        y_2 = y[:,C:2*C]
        y_3 = y[:,2*C:]
        return y_1,y_2,y_3

class Block(nn.Module):
    def __init__(self,dim=512,temperature=3):
        super().__init__()
        self.trans_layer = TransLayer(dim=dim)
        self.mcab = MCAB(dim=dim,temperature=temperature)

    def forward(self,x):
        x = self.trans_layer(x)
        x = self.mcab(x)

        return x

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Attn_Net(nn.Module):

    def __init__(self, L=512, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class CAMIL(MIL):
    """Channel-attention MIL adapted from maojy0914/CAMIL.

    Source: models/CAMIL.py, commit 5481d32a59b0146d9a96f1129bb12ad493acb81a.
    The upstream blocks and initialization are retained. Padded bags are
    processed separately so padding does not enter the spatial/channel blocks.
    """

    def __init__(self, in_dim=1024, embed_dim=512, num_classes=2,
                 temperature=3, dropout=0.25, n_layers=4, gate=True, task_type='classification'):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes, task_type=task_type)
        self._fc1 = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.n_classes = num_classes
        self.layers = nn.ModuleList([
            Block(dim=embed_dim, temperature=temperature) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        attention_cls = Attn_Net_Gated if gate else Attn_Net
        self.attention_net = attention_cls(L=embed_dim, D=embed_dim // 2,
                                          dropout=False, n_classes=1)
        self._fc2 = nn.Linear(embed_dim, num_classes)
        initialize_weights(self)

    def forward_attention(self, h, attn_mask=None, attn_only=True):
        h = self.ensure_batched(h)
        if attn_mask is None:
            attn_mask = torch.ones(h.shape[:2], device=h.device, dtype=torch.bool)
        else:
            attn_mask = attn_mask.to(device=h.device, dtype=torch.bool)
        if attn_mask.shape != h.shape[:2] or not attn_mask.any(dim=1).all():
            raise ValueError('CAMIL requires at least one valid patch per bag')
        embedded, attention = [], []
        for features, valid in zip(h, attn_mask):
            x = self._fc1(features[valid].unsqueeze(0))
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x).squeeze(0)
            scores, _ = self.attention_net(x)
            padded = x.new_zeros((h.shape[1], self.embed_dim))
            padded[valid] = x
            weights = scores.new_full((h.shape[1], 1), torch.finfo(scores.dtype).min)
            weights[valid] = scores
            embedded.append(padded)
            attention.append(weights.transpose(0, 1))
        h, attention = torch.stack(embedded), torch.stack(attention)
        if attn_only:
            return attention
        return h, attention

    def forward_features(self, h, attn_mask=None, return_attention=True):
        h, attention = self.forward_attention(h, attn_mask=attn_mask, attn_only=False)
        features = torch.bmm(F.softmax(attention, dim=-1), h).squeeze(1)
        return features, {'attention': attention if return_attention else None}

    def forward_head(self, h):
        return self._fc2(h)

    def initialize_classifier(self, num_classes=None):
        if num_classes is not None:
            self.num_classes = self.n_classes = num_classes
        self._fc2 = nn.Linear(self.embed_dim, self.num_classes)
        initialize_weights(self._fc2)

    def forward(self, h, loss_fn=None, label=None, attn_mask=None,
                return_attention=False, return_slide_feats=False, censorship=None):
        if self.task_type == 'survival':
            return self.forward_survival(h, label, censorship, loss_fn, attn_mask,
                                         return_attention, return_slide_feats)
        features, logs = self.forward_features(
            h, attn_mask=attn_mask, return_attention=return_attention)
        logits = self.forward_head(features)
        loss = self.compute_loss(loss_fn, logits, label)
        logs['loss'] = loss.item() if loss is not None else -1
        if return_slide_feats:
            logs['slide_feats'] = features
        return {'logits': logits, 'loss': loss}, logs


class CAMILConfig(PretrainedConfig):
    """Hugging Face configuration for CAMIL."""

    model_type = 'camil'

    def __init__(self, in_dim=1024, embed_dim=512, num_classes=2,
                 temperature=3, dropout=0.25, n_layers=4, gate=True, **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.temperature = temperature
        self.dropout = dropout
        self.n_layers = n_layers
        self.gate = gate


class CAMILModel(PreTrainedModel):
    config_class = CAMILConfig

    def __init__(self, config, **kwargs):
        for key, value in kwargs.items():
            setattr(config, key, value)
        super().__init__(config)
        self.model = CAMIL(
            in_dim=config.in_dim, embed_dim=config.embed_dim,
            num_classes=config.num_classes, temperature=config.temperature,
            dropout=config.dropout, n_layers=config.n_layers, gate=config.gate,
            task_type=getattr(config, 'task_type', 'classification')
        )
        config.task_type = self.model.task_type
        config.n_bins = self.model.n_bins
        self.forward = self.model.forward
        self.forward_attention = self.model.forward_attention
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.initialize_classifier = self.model.initialize_classifier


AutoConfig.register(CAMILConfig.model_type, CAMILConfig)
AutoModel.register(CAMILConfig, CAMILModel)
