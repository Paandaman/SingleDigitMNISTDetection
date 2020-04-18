import os

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, datasets
from torch.nn import functional as F

from randaugment import policies as found_policies
from randaugment import augmentation_transforms


if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda")
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = torch.device("cpu")


def supervised_batch(model, batch, eta):
    x, y = batch
    y_ = model(x)
    pred = F.softmax(y_, dim=-1)
    filtered_loss = training_signal_annealing(pred, y, eta)
    return filtered_loss


def unsupervised_batch(model, batch):
    x, _ = batch
    with torch.no_grad():
        y_ = model(x)
    x_augm = random_augmentation(x).float()
    y_augm = model(x_augm)
    kl = _kl_divergence_with_logits(y_, y_augm)
    kl = torch.mean(kl)
    return kl


def random_augmentation(x):
    aug_policies = found_policies.randaug_policies()
    x_augm = torch.zeros_like(x)
    for i in range(x.size()[0]):
        chosen_policy = aug_policies[np.random.choice(
            len(aug_policies))]
        aug_image = augmentation_transforms.apply_policy(
            chosen_policy, x[i,:,:,:].permute(1,2,0).cpu().numpy())
        aug_image = augmentation_transforms.cutout_numpy(aug_image)
        tmp = torch.tensor(aug_image).permute(2,0,1)
        x_augm[i,:,:,:] = torch.mean(tmp, dim=0).unsqueeze(0)
    return x_augm


def _kl_divergence_with_logits(p_logits, q_logits):
  p = F.softmax(p_logits)
  log_p = F.log_softmax(p_logits)
  log_q = F.log_softmax(q_logits)
  kl = torch.sum(p * (log_p - log_q), -1)
  return kl


def training_signal_annealing(pred, ground_truth, eta):
    onehot = F.one_hot(ground_truth, num_classes=pred.size(1)).float()
    correct_label_probs = torch.sum(pred*onehot, -1)
    smaller_than_threshold = torch.lt(correct_label_probs, eta).float()
    smaller_than_threshold.requires_grad = False
    Z = np.maximum(torch.sum(smaller_than_threshold.cpu()), 1).float()
    masked_loss = torch.log(correct_label_probs)*smaller_than_threshold 
    loss = torch.sum(-masked_loss)
    return loss/Z

def get_next_batch(data_iter, data_loader):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        batch = next(data_iter)
    batch = batch[0].to(device), batch[1].to(device)
    return batch, data_iter


def update_eta(schedule, T, k, step):
    if schedule == 'linear' or schedule == None:
        return (step/T)*(1 - 1/k) + 1/k
    elif schedule == 'exponential':
        scale = 5
        step_ratio = step/T
        coeff = np.exp((step_ratio - 1) * scale)
        start = 1./k 
        end = 1
        return coeff * (end - start) + start