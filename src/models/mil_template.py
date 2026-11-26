import torch
import torch.nn as nn
from typing import Optional
from abc import ABC, abstractmethod
from torch.nn import functional as F


class NLLSurvLoss(nn.Module):
    """Patch-GCN discrete survival NLL, evaluated stably from raw logits.

    label is the observed time-bin index (0..K-1); censorship is 0 for an
    observed event, 1 for right censoring. alpha=0 is the ordinary likelihood.
    """

    def __init__(self, alpha=0.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits, label, censorship):
        label = torch.as_tensor(label, device=logits.device).reshape(-1)
        censorship = torch.as_tensor(censorship, device=logits.device).reshape(-1)
        scores = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
        log_survival = F.logsigmoid(-scores).cumsum(dim=1)
        before = F.pad(log_survival, (1, 0))
        y = label.long()[:, None]
        event = -(before.gather(1, y) + F.logsigmoid(scores).gather(1, y))
        censored = -log_survival.gather(1, y)
        c = censorship.to(scores.dtype)[:, None]
        return ((1 - c) * event + (1 - self.alpha) * c * censored).mean()


class MIL(ABC, nn.Module):
    """
    Abstract base class for MIL (Multiple Instance Learning) models.
    Defines the core forward pass methods that MIL implementations must provide.
    """

    def __init__(self,
                 in_dim: int,
                 embed_dim: int,
                 num_classes: int,
                 task_type: str = 'classification',
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.task_type = task_type
        self.n_bins = num_classes if task_type == 'survival' else None

    def survival_logits(self, h, return_attention=False, return_slide_feats=False):
        """One unpadded bag; subclasses may preserve auxiliary prediction streams."""
        features, logs = self.forward_features(h, return_attention=return_attention)
        if not isinstance(logs, dict):
            logs = {'attention': logs}
        logs = {'attention': logs.get('attention')} if return_attention else {}
        if return_slide_feats:
            logs['slide_feats'] = features
        return self.forward_head(features).reshape(1, self.num_classes), logs

    def compute_survival_loss(self, result, label, censorship, loss_fn, bag_logs):
        """Bag-level objective; models override this to add their auxiliary losses."""
        result['loss'] = loss_fn(result['logits'], label, censorship)
        return {}

    def forward_survival(self, h, label=None, censorship=None, loss_fn=None,
                         attn_mask=None, return_attention=False, return_slide_feats=False):
        """Shared censored-survival path, preserving each architecture's bag semantics.

        Bags are processed separately so padding cannot affect spatial/instance
        selection, including in models whose original pooling supports only B=1.
        Optional attention/features are lists with one entry per unpadded bag.
        A custom loss uses the signature loss_fn(logits, label, censorship).
        """
        if h.ndim == 2:
            h = h.unsqueeze(0)
        if attn_mask is None:
            attn_mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        else:
            attn_mask = torch.as_tensor(attn_mask, device=h.device)
            if h.shape[0] == 1 and attn_mask.ndim == 1:
                attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask.bool()
        predictions, bag_logs = [], []
        for bag, valid in zip(h, attn_mask):
            logits, logs = self.survival_logits(bag[valid].unsqueeze(0),
                                               return_attention, return_slide_feats)
            predictions.append(logits)
            bag_logs.append(logs)
        logits = torch.cat(predictions, dim=0)
        hazards = logits.sigmoid()
        survival = torch.cumprod(1 - hazards, dim=1)
        result = dict(logits=logits, hazards=hazards, survival=survival,
                      risk=-survival.sum(dim=1), loss=None)
        logs = {}
        for key in ('attention', 'slide_feats'):
            if any(key in item for item in bag_logs):
                logs[key] = [item.get(key) for item in bag_logs]
        if label is not None:
            objective = NLLSurvLoss() if loss_fn is None else loss_fn
            logs.update(self.compute_survival_loss(result, label, censorship, objective, bag_logs))
        logs['loss'] = result['loss'].item() if result['loss'] is not None else -1
        return result, logs

    @abstractmethod
    def forward_attention(self, h: torch.Tensor)  -> torch.Tensor:
        """
        Method to compute attention scores.

        Args:
            h: [B x M x D]-dim torch.Tensor representing patch embeddings.
            : Optional attention mask.
            attn_only: If True, returns only attention scores; otherwise, returns embeddings and scores.

        Returns:
            A: [B x num. attention heads x M]-dim torch.Tensor (attention scores)
               or (h_transformed, A) if attn_only is False.
        """
        pass

    @abstractmethod
    def forward_features(self, h: torch.Tensor, return_attention: bool=False) -> tuple[torch.Tensor, dict]:
        """
        Aggregate patch features using attention into slide-level features.

        Args:
            h: [B x M x D]-dim torch.Tensor representing patch embeddings.
            return_attention: bool indicating whether to return attention scores in intermed dict.

        Returns:
            h: [B x D]-dim torch.Tensor, the aggregated bag-level feature.
            intermeds: dict containing intermediate results (optional, can be extended by concrete implementations).
        """
        pass

    @abstractmethod
    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply classification head to the aggregated slide-level feature.

        Args:
            h: [B x D]-dim torch.Tensor, the aggregated bag-level feature.

        Returns:
            logits: [B x num_classes]-dim torch.Tensor, the classification logits.
        """
        pass

    @abstractmethod
    def forward(self, h: torch.Tensor,
                loss_fn: nn.Module = False,
                label: torch.LongTensor = None,
                attn_mask=None,
                return_attention: bool = False,
                return_slide_feats: bool = False) -> tuple[dict, dict]:
        """
        Complete forward pass of the model

        Args:
            h: [B x M x D]-dim torch.Tensor representing patch embeddings.
            loss_fn: Optional loss function.
            label: Optional ground truth labels.
            attn_mask: Optional attention mask.
            return_attention: If True, return attention scores in log_dict.
            return_slide_feats: If True, return 'slide_feats' in log_dict.

        Returns:
            results_dict: A dictionary containing the 'logits' and 'loss'
            log_dict: An dictionary optionally containing attention and intermediate results
        """
        pass

    @staticmethod
    def ensure_batched(tensor: torch.Tensor, return_was_unbatched: bool = False) -> torch.Tensor:
        """
        Ensure the tensor is batched (has a batch dimension).

        Args:
            tensor: A torch.Tensor that may or may not have a batch dimension.

        Returns:
            A batched torch.Tensor.
        """
        was_unbatched = False
        while len(tensor.shape) < 3:
            tensor = tensor.unsqueeze(0)
            was_unbatched = True
        if return_was_unbatched:
            return tensor, was_unbatched
        return tensor

    @staticmethod
    def ensure_unbatched(tensor: torch.Tensor, return_was_batched: bool = False) -> torch.Tensor:
        """
        Ensure the tensor is unbatched (removes the batch dimension if present).

        Args:
            tensor: A torch.Tensor that may or may not have a batch dimension.

        Returns:
            An unbatched torch.Tensor.
        """
        was_batched = True
        while len(tensor.shape) > 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
            was_batched = False
        if return_was_batched:
            return tensor, was_batched
        return tensor

    @staticmethod
    def compute_loss(loss_fn: nn.Module,
                     logits: torch.Tensor,
                     label: torch.LongTensor) -> torch.Tensor:
        """
        Compute the loss using the provided loss function.

        Args:
            loss_fn: A callable loss function.
            logits: The model's output logits.
            label: The ground truth labels.

        Returns:
            A scalar tensor representing the computed loss.
        """
        if loss_fn is None or logits is None:
            return None
        return loss_fn(logits, label)

    def initialize_weights(self):
        """
        Initialize the weights of the model with kaiming he for linear layers, and xavier for all others
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)


    def initialize_classifier(self, num_classes: Optional[int] = None):
        """
        Initialize the classifier layer
        """
        if num_classes is None:
            num_classes = self.num_classes
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        nn.init.kaiming_uniform_(self.classifier.weight, nonlinearity='relu')
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)





