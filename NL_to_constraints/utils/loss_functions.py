from typing import Union, List
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment as lsa


logger = logging.getLogger(__name__)


def axe_loss(logits: torch.FloatTensor,
             logit_lengths: torch.Tensor,
             targets: torch.LongTensor,
             target_lengths: torch.Tensor,
             blank_index: int,
             delta: float,
             reduction: str = 'mean',
             label_smoothing: float = None,
             return_a: bool = False
             ) -> Union[torch.FloatTensor, List[torch.Tensor]]:
    """Aligned Cross Entropy
    Marjan Ghazvininejad, Vladimir Karpukhin, Luke Zettlemoyer, Omer Levy, in arXiv 2020
    https://arxiv.org/abs/2004.01655
    Computes the aligned cross entropy loss with parallel scheme.
    Parameters
    ----------
    logits : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    logit_lengths : ``torch.Tensor``, required.
        A ``torch.Tensor`` of size (batch_size,)
        which contains lengths of the logits
    targets : ``torch.LongTensor``, required.
        A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    target_lengths : ``torch.Tensor``, required.
        A ``torch.Tensor`` of size (batch_size,)
        which contains lengths of the targets
    blank_index : ``int``, required.
        An index of special blank token.
    delta : ``float``, required.
        A ``float`` for penalizing skip target operators.
    reduction : ``str``, optional.
        Specifies the reduction to apply to the output.
        Default "mean".
    label_smoothing : ``float``, optional
        Whether or not to apply label smoothing.
    return_a : ``bool``, optional.
        Whether to return the matrix of conditional axe values. Default is False.
    """
    assert targets.size(0) == logits.size(
        0), f'Inconsistency of batch size,  {targets.size(0)} of targets and {logits.size(0)} of logits.'

    batch_size, logits_sequence_length, num_class = logits.shape
    _, target_sequence_length = targets.shape
    device = logits.device

    # for torch.gather
    targets = targets.unsqueeze(-1)  # batch_size, target_sequence_length, 1

    # (batch_size, target_sequence_length + 1, logits_sequence_length + 1)
    batch_A = torch.zeros(targets.size(0), targets.size(1) + 1, logits.size(1) + 1).to(device)
    batch_blank_index = torch.full((logits.size(0), 1), blank_index, dtype=torch.long).to(device)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # i -> Targets
    # j -> Predictions
    # A_{i,0} = A_{i−1,0} − delta * log P_1 (Y_i)
    for i in range(1, targets.size(1) + 1):
        # batch_A[:, i, 0] is calculated from targets[:, i-1, :], because batch_A added 0-th row
        batch_A[:, i, 0] = batch_A[:, i - 1, 0] - delta * torch.gather(log_probs[:, 0, :], dim=1,
                                                                       index=targets[:, i - 1, :]).squeeze(-1)

    # A_{0,j} = A_{0,j−1} − log P_j ("BLANK")
    for j in range(1, logits.size(1) + 1):
        # batch_A[:, 0, j] is calculated from log_probs[:, j-1, :], because batch_A added 0-th column
        batch_A[:, 0, j] = batch_A[:, 0, j - 1] - delta * torch.gather(log_probs[:, j - 1, :], dim=1,
                                                                       index=batch_blank_index).squeeze(-1)

    # flip logit dim to get anti-diagonal part by using use torch.diag
    # flip on sequence length dimension
    batch_A_flip = batch_A.flip(-1)  # (batch_size, target_sequence_length + 1, logits_sequence_length + 1)
    log_probs_flip = log_probs.flip(-2)  # (batch_size, sequence_length, num_classes)

    # to extract indices for the regions corresponding diag part.
    # create a 2D tensor with each row being torch.arange(logits.size(1))
    map_logits = torch.arange(logits.size(1)) - torch.zeros(targets.size(1), 1)
    map_targets = torch.arange(targets.size(1)).unsqueeze(-1) - torch.zeros((1, logits.size(1)))
    # index must be int
    map_logits = map_logits.long().to(device)
    map_targets = map_targets.long().to(device)

    for i in range(logits.size(1) - 1, -targets.size(1), -1):

        # batch_A_flip_sets[:, :, :, 0] : batch_A[:, i  , j-1]
        # batch_A_flip_sets[:, :, :, 1] : batch_A[:, i-1, j  ]
        # batch_A_flip_sets[:, :, :, 2] : batch_A[:, i-1, j-1]

        batch_A_flip_sets = torch.cat((batch_A_flip.roll(shifts=-1, dims=-1).unsqueeze(-1),
                                       batch_A_flip.roll(shifts=1, dims=-2).unsqueeze(-1),
                                       batch_A_flip.roll(shifts=(1, -1), dims=(-2, -1)).unsqueeze(-1)),
                                      dim=-1)
        # batch_A_flip_sets : [batch_size, sequence_length, num_classes, 3]
        # trimming - Ignore the first row and column in the original matrix (batch_A)
        # - the last column (A_{0,j} = A_{0,j−1} − log P_j ("BLANK"))
        # - the first row (A_{i,0} = A_{i−1,0} − delta * log P_1 (Y_i))
        batch_A_flip_sets_trim = batch_A_flip_sets[:, 1:, :-1, :]

        # extracting anti-diagonal part
        # (batch, 3, num_diag)
        A_diag = batch_A_flip_sets_trim.diagonal(offset=i, dim1=-3, dim2=-2)

        # (batch, num_diag, 3)
        A_diag = A_diag.transpose(-1, -2)
        num_diag = A_diag.size(1)
        logit_indices = map_logits.diagonal(offset=i, dim1=-2, dim2=-1)
        # log_probs_diag : (batch, num_diag, num_class)
        # Choose the slice corresponding to the values of j as per the diagonal
        log_probs_flip_diag = log_probs_flip[:, logit_indices[0]:logit_indices[-1] + 1, :]
        target_indices = map_targets.diagonal(offset=i, dim1=-2, dim2=-1)
        # targets_diag : (batch, num_diag, num_class)
        # Choose the slice corresponding to the values of i as per the diagonal
        targets_diag = targets[:, target_indices[0]:target_indices[-1] + 1, :]

        # align, skip_prediction, skip_target
        batch_align = A_diag[:, :, 2] - torch.gather(log_probs_flip_diag, dim=2, index=targets_diag).squeeze(-1)
        batch_skip_prediction = A_diag[:, :, 0] - torch.gather(log_probs_flip_diag, dim=2,
                                                               index=batch_blank_index.expand(-1, num_diag).unsqueeze(
                                                                   -1)).squeeze(-1)
        batch_skip_target = A_diag[:, :, 1] - delta * torch.gather(log_probs_flip_diag, dim=2,
                                                                   index=targets_diag).squeeze(-1)

        # (batch_size, num_diag, 3)
        operations = torch.cat(
            (batch_align.unsqueeze(-1), batch_skip_prediction.unsqueeze(-1), batch_skip_target.unsqueeze(-1)), dim=-1)

        # (batch_size, num_diag)
        diag_axe = torch.min(operations, dim=-1).values

        assert logits.size(1) >= targets.size(1), "assuming target length =< logit length."
        #Continue checking from this point onwards
        # Put computed values back in the appropriate locations in the batch_A_flip matrix
        if i > (logits.size(1) - targets.size(1)):
            # (batch_size, logits_length, logits_length)
            # -> (batch_size, targets_length, logits_length)
            axe = torch.diag_embed(diag_axe, offset=i, dim1=-2, dim2=-1)
            batch_A_flip[:, 1:, :-1] += axe[:, :targets.size(1), :]
        elif i > 0:
            # (batch_size, logits_length, logits_length)
            # -> (batch_size, targets_length, logits_length)
            axe = torch.diag_embed(diag_axe, offset=0, dim1=-2, dim2=-1)
            batch_A_flip[:, 1:, i: i + targets.size(1)] += axe
        else:
            axe = torch.diag_embed(diag_axe, offset=i, dim1=-2, dim2=-1)
            batch_A_flip[:, 1:, :targets.size(1)] += axe

    # recover correct order in logit dim
    batch_A = batch_A_flip.flip(-1)

    # rm 0-th row and column
    _batch_A = batch_A[:, 1:, 1:]

    ## Gather A_nm, avoiding masks
    # index_m : (batch_size, target_sequence_length, 1)
    index_m = logit_lengths.unsqueeze(-1).expand(-1, _batch_A.size(1)).unsqueeze(-1).long().to(logits.device)

    # gather m-th colmun
    # batch_A_nm : (batch_size, target_sequence_length, 1)
    # index_m occors out of bounds for index
    batch_A_m = torch.gather(_batch_A, dim=2, index=(index_m - 1))
    batch_A_m = batch_A_m.squeeze(-1)

    # index_n : (batch_size, 1)
    index_n = target_lengths.unsqueeze(-1).long().to(logits.device)

    # gather n-th row
    # batch_A_nm : (batch_size, 1, 1)
    batch_A_nm = torch.gather(batch_A_m, dim=1, index=(index_n - 1))

    # batch_A_nm : (batch_size)
    batch_A_nm = batch_A_nm.squeeze(-1)

    if reduction == "mean":
        axe_nm = batch_A_nm.mean()
    else:
        raise NotImplementedError

    # Refs fairseq nat_loss.
    # https://github.com/pytorch/fairseq/blob/6f6461b81ac457b381669ebc8ea2d80ea798e53a/fairseq/criterions/nat_loss.py#L70
    # actuary i'm not sure this is reasonable.
    if label_smoothing is not None and label_smoothing > 0.0:
        axe_nm = axe_nm * (1.0 - label_smoothing) - log_probs.mean() * label_smoothing

    if return_a:
        return axe_nm, batch_A.detach()

    return axe_nm

def oaxe(logits: torch.FloatTensor,
         targets: torch.LongTensor,
         ce_criterion: torch.nn.CrossEntropyLoss):
    negative_log_probabilities = -F.log_softmax(logits, dim = 2)
    '''
    Order Agnostic Cross Entropy 
    https://arxiv.org/pdf/2106.05093.pdf
    Parameters
    ----------- 
    logits: ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets: ``torch.LongTensor``, required.
        A ``torch.Tensor`` of size (batch_size,)
        which contains lengths of the logits
    ce_criterion: ``torch.nn.CrossEntropyLoss``, required.
        A pytorch cross entropy function
    '''
    targets = targets.unsqueeze(-1)
    device = logits.device
    weights = torch.zeros(targets.size(0), logits.size(1), targets.size(1)).to(device)
    # Create weight matrix where each weights[:,i,j] corresponds to the probability of predicting target j at
    # position i for each batch
    for i in range(targets.shape[1]):
        weights[:,i,:] = torch.gather(negative_log_probabilities[:,i,:], dim=1, index=targets[:,:,0])
    weights = weights.detach().cpu().numpy()
    best_match = np.repeat(np.arange(targets.shape[1]).reshape(1, -1, 1), targets.shape[0], axis=0)

    # Populate best match array [bs, num_constraints, 1]
    # For each batch best_match[b,i,0] consists of the index of the target label chosen for the ith position
    # in the sequence for batch b.
    for b in range(weights.shape[0]):
        pred_ind, tar_ind = lsa(weights[b,:,:])
        best_match[b,:] = tar_ind.reshape(-1,1)
    targets = targets.squeeze(-1)
    logits = logits.view(-1, logits.shape[-1])
    best_match = torch.Tensor(best_match).to(device).long()
    best_match = best_match.squeeze(-1)

    # Collect the actual target labels corresponding to the order of indices in best_match
    aligned_targets = torch.gather(targets, dim = 1, index=best_match)
    aligned_targets = aligned_targets.view(-1)
    loss = ce_criterion(logits, aligned_targets)

    return loss
