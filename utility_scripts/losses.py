import math
import scipy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#################################################

#   Age Estimation deep label distribution loss

#################################################

def label_to_gaussian(label, min_age, max_age, sigma):

    ages = torch.arange(min_age, max_age + 1, device=label.device).float().view(1, -1)
    label = label.view(-1, 1).float()
    
    # Calculate Gaussian
    squared_diff = torch.pow(ages - label, 2)
    distribution = torch.exp(-squared_diff / (2 * sigma ** 2))
    
    # Normalize with epsilon for safety
    distribution = distribution / (torch.sum(distribution, dim=1, keepdim=True) + 1e-10)
    
    return distribution

def dldl_loss(logits, target, min_age=0, max_age=101, sigma=2.0, l1_weight=1.0):
    # 1. Calc Probabilities
    log_probs = F.log_softmax(logits, dim=1) 
    probs = torch.exp(log_probs)
    
    # 2. Target Dist
    target_distribution = label_to_gaussian(target, min_age, max_age, sigma)
    
    # 3. KL Loss
    loss_kl = F.kl_div(log_probs, target_distribution, reduction='batchmean')
    
    # 4. Expectation L1
    # Create ages vector consistent with logits dimension
    ages = torch.arange(min_age, max_age + 1, device=logits.device).float()
    
    # Verify shape consistency (Optional debugging)
    if logits.size(1) != len(ages):
        raise ValueError(f"Logits classes {logits.size(1)} != Age range {len(ages)}")

    expectation = torch.sum(probs * ages, dim=1)
    loss_l1 = F.l1_loss(expectation, target.float())
    
    return loss_kl + (l1_weight * loss_l1)


#################################################

#   Pose estimation geodesic loss

#################################################

class GeodesicLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, m1, m2):
        
        R_diffs = m1 @ m2.permute(0, 2, 1)
        
        traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        
        cos = torch.clamp((traces - 1.0) / 2.0, -1.0 + self.eps, 1.0 - self.eps)
        radians = torch.acos(cos)
        
        return radians.mean()



import torch
import torch.nn as nn

class AsymmetricLossOptimized(nn.Module):
    def __init__(self, gamma_neg=3, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False, label_smoothing=0.05):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.label_smoothing = label_smoothing

    def forward(self, x, y):
        # 1. SMOOTH LABELS: Strictly for the base Binary Cross Entropy math
        targets_smooth = y * (1 - self.label_smoothing * 2) + self.label_smoothing
        anti_targets_smooth = 1.0 - targets_smooth

        # 2. HARD LABELS: Strictly for routing the Asymmetric Focal weights
        targets_hard = y
        anti_targets_hard = 1.0 - y

        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        # Asymmetric Clipping (only applies to the negative class predictions)
        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

        # 3. Base Cross Entropy Loss (Using SMOOTH targets)
        loss = targets_smooth * torch.log(xs_pos.clamp(min=self.eps))
        loss.add_(anti_targets_smooth * torch.log(xs_neg.clamp(min=self.eps)))

        # 4. Asymmetric Focal Scaling (Using HARD targets to maintain exact focal math)
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            
            # Isolate p_t using hard 1s and 0s
            pt_pos = xs_pos * targets_hard
            pt_neg = xs_neg * anti_targets_hard
            
            # Calculate (1 - p_t)^gamma
            asymmetric_w = torch.pow(
                1 - pt_pos - pt_neg,
                self.gamma_pos * targets_hard + self.gamma_neg * anti_targets_hard
            )
            
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
                
            loss *= asymmetric_w

        return -loss.sum() / x.size(0)



class WingLoss(nn.Module):
    def __init__(self, w=15.0, epsilon=3.0, image_size = 112):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.image_size = image_size
        self.c = self.w - self.w * math.log(1 + self.w / self.epsilon)


    def forward(self, predictions, targets, mask = None):
        
        # scale the coordinates to be between 0 and self.image_size instead of between 0 and 1.
        predictions = predictions * self.image_size
        targets = targets * self.image_size
        
        # predictions and targets should be shape (Batch, 136)
        x = torch.abs(predictions - targets)
        
        # Where error is small (x < w), apply logarithmic loss
        loss1 = self.w * torch.log(1 + x / self.epsilon)
        
        # Where error is large (x >= w), apply linear L1 loss
        loss2 = x - self.c
        
        # Combine them
        loss = torch.where(x < self.w, loss1, loss2)
        
        if mask != None:
            mask_expanded = torch.repeat_interleave(mask, repeats=2, dim=1)
            loss = loss * mask_expanded
        
        return loss.mean()


class SoftWingLoss(nn.Module):
    def __init__(self, w1 = 2, w2 = 15, epsilon=0.5, image_size = 112):
        super(SoftWingLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.epsilon = epsilon
        self.B = self.w1 - self.w2 * math.log(1 + self.w1 / self.epsilon)
        self.image_size = image_size

    def forward(self, predictions, targets, mask = None):
        
        # scale the coordinates to be between 0 and self.image_size instead of between 0 and 1.
        predictions = predictions * self.image_size
        targets = targets * self.image_size
        
        # predictions and targets should be shape (Batch, 136)
        x = torch.abs(predictions - targets)
        
        # Where error is small (x < w1), apply linear loss
        loss1 = x
        
        # Where error is large (x >= w1), apply logarithmic loss
        loss2 = self.w2 * torch.log(1 + x / self.epsilon) + self.B
        
        # Combine them
        loss = torch.where(x < self.w1, loss1, loss2)
        
        if mask != None:
            mask_expanded = torch.repeat_interleave(mask, repeats=2, dim=1)
            loss = loss * mask_expanded
        
        return loss.mean()


class FocalLoss(nn.Module): # for emotion recognition
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[target] * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
