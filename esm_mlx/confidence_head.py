# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Confidence scoring mechanism (pLDDT) for ESMFold structure prediction."""

import math
from typing import Dict, Tuple

import mlx.core as mx
import mlx.nn as nn


class CategoricalMixture:
    """
    Categorical mixture distribution for confidence scoring.
    
    Models confidence scores as a categorical distribution over bins,
    which is more stable than direct regression.
    """
    
    def __init__(self, logits: mx.array, bins: int = 50, start: float = 0.0, end: float = 1.0):
        """
        Args:
            logits: [batch, N_res, bins] - logits for categorical distribution
            bins: Number of bins for the distribution
            start: Start value for the range
            end: End value for the range  
        """
        self.logits = logits
        
        # Create bin centers
        bin_edges = mx.linspace(start, end, bins + 1)
        self.v_bins = (bin_edges[:-1] + bin_edges[1:]) / 2  # [bins]
        
    def log_prob(self, true_values: mx.array) -> mx.array:
        """
        Compute log probability of true values under the categorical mixture.
        
        Args:
            true_values: [batch, N_res] - ground truth values
            
        Returns:
            [batch, N_res] - log probabilities
        """
        # Find closest bin for each true value
        # true_values: [batch, N_res] -> [batch, N_res, 1]
        # v_bins: [bins] -> [1, 1, bins]
        expanded_true = true_values[..., None]  # [batch, N_res, 1]
        expanded_bins = self.v_bins[None, None, :]  # [1, 1, bins]
        
        # Find the bin with minimum distance
        distances = mx.abs(expanded_true - expanded_bins)  # [batch, N_res, bins]
        true_indices = mx.argmin(distances, axis=-1)  # [batch, N_res]
        
        # Get log probabilities
        log_probs = mx.log_softmax(self.logits, axis=-1)  # [batch, N_res, bins]
        
        # Select probabilities for the true bins
        batch_size, N_res = true_indices.shape
        batch_indices = mx.arange(batch_size)[:, None]  # [batch, 1]
        res_indices = mx.arange(N_res)[None, :]  # [1, N_res]
        
        selected_log_probs = log_probs[batch_indices, res_indices, true_indices]
        
        return selected_log_probs
    
    def mean(self) -> mx.array:
        """
        Compute expected value of the categorical mixture.
        
        Returns:
            [batch, N_res] - expected confidence scores
        """
        probs = mx.softmax(self.logits, axis=-1)  # [batch, N_res, bins]
        
        # Weighted sum: E[X] = sum_i p_i * x_i
        # probs: [batch, N_res, bins]
        # v_bins: [bins] -> [1, 1, bins]
        expanded_bins = self.v_bins[None, None, :]
        expected_values = mx.sum(probs * expanded_bins, axis=-1)  # [batch, N_res]
        
        return expected_values


class LDDTHead(nn.Module):
    """
    Local Distance Difference Test (LDDT) confidence prediction head.
    
    Predicts per-residue confidence scores using a categorical distribution
    over confidence bins, which provides more stable training than regression.
    """
    
    def __init__(self, c_s: int, num_bins: int = 50):
        super().__init__()
        
        self.c_s = c_s
        self.num_bins = num_bins
        
        # Multi-layer network for confidence prediction
        self.confidence_net = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s // 2),
            nn.ReLU(),
            nn.Linear(c_s // 2, num_bins)
        )
        
    def __call__(self, s: mx.array) -> CategoricalMixture:
        """
        Args:
            s: [batch, N_res, c_s] - single representation
            
        Returns:
            CategoricalMixture object for confidence prediction
        """
        logits = self.confidence_net(s)  # [batch, N_res, num_bins]
        return CategoricalMixture(logits, bins=self.num_bins, start=0.0, end=1.0)


class DistogramHead(nn.Module):
    """
    Distance prediction head for computing inter-residue distances.
    
    Predicts distance distributions which can be used for contact prediction
    and structure validation.
    """
    
    def __init__(self, c_z: int, num_bins: int = 64, min_dist: float = 0.0, max_dist: float = 20.0):
        super().__init__()
        
        self.c_z = c_z
        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        
        # Distance prediction network
        self.distance_net = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, num_bins)
        )
        
        # Create distance bins
        self.distance_bins = mx.linspace(min_dist, max_dist, num_bins)
        
    def __call__(self, z: mx.array) -> Dict[str, mx.array]:
        """
        Args:
            z: [batch, N_res, N_res, c_z] - pair representation
            
        Returns:
            Dictionary containing distance predictions
        """
        logits = self.distance_net(z)  # [batch, N_res, N_res, num_bins]
        
        # Convert to distance distribution
        distance_probs = mx.softmax(logits, axis=-1)
        
        # Expected distances
        expected_distances = mx.sum(
            distance_probs * self.distance_bins[None, None, None, :],
            axis=-1
        )  # [batch, N_res, N_res]
        
        return {
            "distance_logits": logits,
            "distance_probs": distance_probs,
            "expected_distances": expected_distances
        }


class TMScorePredictor(nn.Module):
    """
    Template Modeling Score (TM-score) predictor for global structure quality.
    
    Predicts a single confidence score for the entire structure.
    """
    
    def __init__(self, c_s: int):
        super().__init__()
        
        # Global pooling and prediction network
        self.tm_net = nn.Sequential(
            nn.Linear(c_s, c_s // 2),
            nn.ReLU(),
            nn.Linear(c_s // 2, c_s // 4),  
            nn.ReLU(),
            nn.Linear(c_s // 4, 1),
            nn.Sigmoid()  # TM-score is bounded between 0 and 1
        )
        
    def __call__(self, s: mx.array, mask: mx.array = None) -> mx.array:
        """
        Args:
            s: [batch, N_res, c_s] - single representation
            mask: [batch, N_res] - sequence mask
            
        Returns:
            [batch] - predicted TM-scores
        """
        # Global average pooling with mask
        if mask is not None:
            mask_expanded = mask[..., None]  # [batch, N_res, 1]
            s_masked = s * mask_expanded
            pooled = mx.sum(s_masked, axis=1) / mx.sum(mask, axis=1, keepdims=True)
        else:
            pooled = mx.mean(s, axis=1)  # [batch, c_s]
        
        # Predict TM-score
        tm_score = self.tm_net(pooled).squeeze(-1)  # [batch]
        
        return tm_score


class ConfidenceHead(nn.Module):
    """
    Complete confidence prediction head combining multiple confidence metrics.
    """
    
    def __init__(self, c_s: int, c_z: int, lddt_bins: int = 50, distogram_bins: int = 64):
        super().__init__()
        
        self.lddt_head = LDDTHead(c_s, lddt_bins)
        self.distogram_head = DistogramHead(c_z, distogram_bins)
        self.tm_predictor = TMScorePredictor(c_s)
        
        # Optional: confidence calibration network
        self.calibration_net = nn.Sequential(
            nn.Linear(2, 8),  # Input: pLDDT + TM-score
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def __call__(
        self, 
        s: mx.array, 
        z: mx.array,
        mask: mx.array = None
    ) -> Dict[str, mx.array]:
        """
        Args:
            s: [batch, N_res, c_s] - single representation
            z: [batch, N_res, N_res, c_z] - pair representation  
            mask: [batch, N_res] - sequence mask
            
        Returns:
            Dictionary containing all confidence predictions
        """
        
        # Per-residue confidence (pLDDT)
        lddt_dist = self.lddt_head(s)
        plddt_scores = lddt_dist.mean()  # [batch, N_res]
        
        # Distance predictions
        distance_results = self.distogram_head(z)
        
        # Global confidence (TM-score)
        tm_scores = self.tm_predictor(s, mask)  # [batch]
        
        # Optional: calibrated confidence combining pLDDT and TM-score
        if mask is not None:
            # Average pLDDT over valid residues
            masked_plddt = plddt_scores * mask
            avg_plddt = mx.sum(masked_plddt, axis=1) / mx.sum(mask, axis=1)
        else:
            avg_plddt = mx.mean(plddt_scores, axis=1)
        
        # Combine pLDDT and TM-score for calibrated confidence
        combined_features = mx.stack([avg_plddt, tm_scores], axis=-1)  # [batch, 2]
        calibrated_confidence = self.calibration_net(combined_features).squeeze(-1)  # [batch]
        
        return {
            "plddt": plddt_scores,
            "plddt_logits": lddt_dist.logits,
            "tm_score": tm_scores,
            "calibrated_confidence": calibrated_confidence,
            **distance_results
        }
        

def compute_lddt_loss(
    pred_lddt: CategoricalMixture,
    true_lddt: mx.array,
    mask: mx.array = None
) -> mx.array:
    """
    Compute LDDT prediction loss using the categorical mixture.
    
    Args:
        pred_lddt: CategoricalMixture for predicted LDDT
        true_lddt: [batch, N_res] - true LDDT scores
        mask: [batch, N_res] - sequence mask
        
    Returns:
        Scalar loss value
    """
    log_probs = pred_lddt.log_prob(true_lddt)  # [batch, N_res]
    
    # Apply mask and compute negative log likelihood
    if mask is not None:
        log_probs = log_probs * mask
        loss = -mx.sum(log_probs) / mx.sum(mask)
    else:
        loss = -mx.mean(log_probs)
    
    return loss