import torch
import torch.nn.functional as F

def get_age_expectation(logits, min_age = 0, max_age = 101):
    log_probs = F.log_softmax(logits, dim=1) 
    probs = torch.exp(log_probs)
    ages = torch.arange(min_age, max_age + 1, device=logits.device).float()
    expectation = torch.sum(probs * ages, dim=1)

    return expectation.detach().cpu().numpy()