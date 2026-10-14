"""CE only: each complete record has equal weight, prompts/padding have none."""
import numpy as np
from json_sft.evaluate import loss_sum as loss

OBJECTIVE = 'gold-output-eos-record-mean-metal-ce-v1'


def numpy_record_ce(logits, labels):
    """Independent FP64 reference; use the SAME logits as the traced CE path."""
    values = np.asarray(logits, np.float64)
    labels = np.asarray(labels)
    if values.ndim != 3 or labels.shape != values.shape[:2]:
        raise ValueError('Expected batch/time/vocabulary logits and shifted labels')
    mask = labels != -100
    safe = np.where(mask, labels, 0)
    if ((safe < 0) | (safe >= values.shape[-1])).any():
        raise ValueError('Label outside vocabulary')
    shifted = values - values.max(axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    nll = -np.take_along_axis(logp, safe[..., None], axis=-1)[..., 0]
    return (nll * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
