import torch

def loss_fn(model, x, marginal_prob_std, condition, eps=1e-5):
    """The loss function for training score-based generative models.
    Args:
    model: A PyTorch model instance that represents a
        time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
    eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps #### t 的范围为啥这么小
    z = torch.randn_like(x)
    alpha_t, std = marginal_prob_std(random_t)
    perturbed_x = x * alpha_t[:, None] + z * std[:, None]
    score = model(perturbed_x, random_t, condition=condition)
    loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=(1,))) # sorce_loss
    return loss



