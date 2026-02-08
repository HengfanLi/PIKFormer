import torch
from torch.nn.utils import clip_grad_norm_

def get_optimizer(model, learning_rate):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )
    
    # Wrap optimizer step with gradient clipping
    original_step = optimizer.step
    
    def step_with_clip():
        clip_grad_norm_(model.parameters(), max_norm=50.0)
        original_step()
        
    optimizer.step = step_with_clip
    
    return optimizer