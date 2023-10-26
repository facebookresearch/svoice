# The following piece of code was adapted from https://github.com/kaituoxu/Conv-TasNet
# released under the MIT License.
# Author: Kaituo XU
# Created on 2018/12

from itertools import permutations

import torch
import torch.nn.functional as F

EPS = 1e-8


def cal_loss(source, estimate_source, source_lengths):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    max_snr, perms, max_snr_idx, snr_set = cal_si_snr_with_pit(source,
                                                               estimate_source,
                                                               source_lengths)
    B, C, T = estimate_source.shape
    loss = 0 - torch.mean(max_snr)

    reorder_estimate_source = reorder_source(
        estimate_source, perms, max_snr_idx)
    return loss, max_snr, estimate_source, reorder_estimate_source


def cal_si_snr_with_pit(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source2 = estimate_source*mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source2, dim=2,
                              keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source2 - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target,
                              dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(
        s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(
        pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]
    pair_wise_si_snr = torch.transpose(pair_wise_si_snr, 1, 2)

    # Get max_snr of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    #  max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C
    return max_snr, perms, max_snr_idx, snr_set / C


def reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, T]
    """
    B, C, *_ = source.size()
    # [B, C], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
    # print('max_snr_perm', max_snr_perm)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reorder_source = torch.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source


def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return mask
