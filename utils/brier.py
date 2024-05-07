import torch

def brier_scores(probabilities, targets, n_classes):
    """
    Calculate the Brier score for multi-class predictions using logits.

    Args:
    logits (torch.Tensor): Predicted logits for each class. Shape should be (N, C) where
                           N is the number of samples and C is the number of classes.
    targets (torch.Tensor): Actual class indices. Shape should be (N,)
    n_classes (int): Number of classes.

    Returns:
    float: The Brier score for the given predictions and actual outcomes.
    """
    # Check if logits and targets are on the same device and have compatible shapes
    if probabilities.dim() != 2 or targets.dim() != 1:
        raise ValueError("Logits must be 2D and targets must be 1D.")
    if probabilities.shape[0] != targets.shape[0]:
        raise ValueError("Logits and targets must have the same number of samples.")
    if probabilities.shape[1] != n_classes:
        raise ValueError("The second dimension of logits must be the number of classes.")
    
    if targets.device != probabilities.device:
        targets = targets.to(probabilities.device)

    # Convert class indices to one-hot encoded format
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=n_classes).float()

    # Calculate the squared differences for all class probabilities against the one-hot targets
    squared_differences = (probabilities - targets_one_hot)**2

    # Compute the Brier score
    mean_brier_score = torch.mean(torch.sum(squared_differences, dim=1))
    return mean_brier_score


def xbrier_scores(labels, probs=None, logits=None):
    """Compute elementwise Brier score using PyTorch for GPU support.

    Args:
      labels: Tensor of integer labels shape [N1, N2, ...]
      probs: Tensor of categorical probabilities of shape [N1, N2, ..., M].
      logits: If `probs` is None, class probabilities are computed as a softmax
        over these logits, otherwise, this argument is ignored.
    Returns:
      Tensor of shape [N1, N2, ...] consisting of Brier score contribution from
      each element. The full-dataset Brier score is an average of these values.
    """
    assert (probs is None) != (logits is None)
    if probs is None:
        probs = torch.softmax(logits, dim=-1)
    nlabels = probs.shape[-1]
    flat_probs = probs.view(-1, nlabels)
    flat_labels = labels.view(-1)

    indices = torch.arange(flat_labels.size(0), device=labels.device)
    plabel = flat_probs[indices, flat_labels]
    out = (flat_probs**2).sum(dim=1) - 2 * plabel
    return out.view(labels.shape)