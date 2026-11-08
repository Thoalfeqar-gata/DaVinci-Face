import numpy as np
import torch
import torch.nn as nn

"""

    Weighting by learning speed.

"""

class DynamicWeightAverage:
    def __init__(self, num_tasks, Temperature = 1.0):
        self.num_tasks = num_tasks
        self.T = Temperature
        # Store the average loss of each task from the previous epoch
        self.avg_losses_previous_epoch = None

    def get_weights(self, avg_losses_current_epoch):
        """
            Calculates the dynamic weights for each task based on the current and previous epoch's average losses.
            Args:
                avg_losses_current_epoch: A numpy array of the average loss for each task in the current epoch.
            Returns:
                weights: A numpy array of weights for each task.
        """
        if avg_losses_current_epoch is None:
            return np.ones(self.num_tasks, dtype=np.float32)

        # On the first epoch, use equal weights and store the current losses for the next epoch.
        if self.avg_losses_previous_epoch is None:
            weights = np.ones(self.num_tasks, dtype=np.float32)
            self.avg_losses_previous_epoch = avg_losses_current_epoch.copy()
            return weights
        
        else:
            W = np.zeros(self.num_tasks, dtype=np.float32)
            W = np.divide(avg_losses_current_epoch, self.avg_losses_previous_epoch + 1e-8)
            weights = np.exp(W / self.T)
            weights = weights / np.sum(weights)
            weights *= self.num_tasks

            self.avg_losses_previous_epoch = avg_losses_current_epoch.copy()

            return weights



class UncertaintyLoss(nn.Module):
    """
    Implements Homoscedastic Uncertainty weighting for Multi-Task Learning.
    Reference: 'Multi-Task Learning Using Uncertainty to Weigh Losses' (Kendall et al.)
    """
    def __init__(self, num_tasks=11): 
        super(UncertaintyLoss, self).__init__()

        # Initialized to 0.0 (which means a weight of 1.0 initially)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses, valid_mask=None):
        """
        losses: A list or tensor of your raw task losses.
        valid_mask: A boolean list indicating if the task has > 0 samples in this batch.
        """
        total_loss = 0
        
        for i, raw_loss in enumerate(losses):
            # If the task isn't in this batch, skip it entirely.
            # This prevents PyTorch from pushing log_vars[i] to negative infinity.
            if valid_mask is not None and not valid_mask[i]:
                continue
                
            # Calculate the precision (1 / sigma^2)
            # exp(-log_var) = 1 / exp(log_var)
            precision = torch.exp(-self.log_vars[i])
            
            # Scale the loss
            scaled_loss = precision * raw_loss + self.log_vars[i]
            
            total_loss += scaled_loss
            
        return total_loss

    def get_weights(self):
        return torch.exp(-self.log_vars).detach().cpu().numpy()