from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            predictions = outputs["preds"]
            is_correct = mask & (predictions == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            valid_metrics = new_carry.halted & (loss_counts > 0)
            accuracy_per_seq = torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0)

            batch_size, num_classes = labels.shape[0], outputs["logits"].shape[-1]
            f1_per_seq = torch.zeros(batch_size, dtype=torch.float32, device=labels.device)

            valid_token_mask = mask & valid_metrics.view(-1, 1)
            if valid_token_mask.any():
                batch_ids = torch.arange(batch_size, device=labels.device).unsqueeze(-1).expand_as(labels)
                batch_ids = batch_ids[valid_token_mask]
                true_labels = labels[valid_token_mask]
                pred_labels = predictions[valid_token_mask]

                correct_mask = (true_labels == pred_labels)

                def _bincount_by_batch(batch_indices: torch.Tensor, class_indices: torch.Tensor):
                    if batch_indices.numel() == 0:
                        return torch.zeros(batch_size, num_classes, dtype=torch.float32, device=labels.device)
                    flat_indices = batch_indices.to(torch.int64) * num_classes + class_indices.to(torch.int64)
                    counts = torch.bincount(flat_indices, minlength=batch_size * num_classes)
                    return counts.view(batch_size, num_classes).to(torch.float32)

                tp = _bincount_by_batch(batch_ids[correct_mask], true_labels[correct_mask])
                fp = _bincount_by_batch(batch_ids[~correct_mask], pred_labels[~correct_mask])
                fn = _bincount_by_batch(batch_ids[~correct_mask], true_labels[~correct_mask])

                precision = torch.zeros_like(tp)
                denom = tp + fp
                precision = torch.where(denom > 0, tp / denom, precision)

                recall = torch.zeros_like(tp)
                denom = tp + fn
                recall = torch.where(denom > 0, tp / denom, recall)

                f1_per_class = torch.zeros_like(tp)
                denom = precision + recall
                f1_per_class = torch.where(denom > 0, 2 * precision * recall / denom, f1_per_class)

                class_support = (tp + fp + fn) > 0
                support = class_support.sum(-1).clamp_min(1)
                f1_per_seq = torch.where(
                    valid_metrics,
                    (f1_per_class * class_support.to(torch.float32)).sum(-1) / support,
                    torch.zeros_like(f1_per_seq),
                )

            # Metrics (halted)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": accuracy_per_seq.sum(),
                "accuracy_sq": (accuracy_per_seq.square()).sum(),
                "f1": torch.where(valid_metrics, f1_per_seq, 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()
