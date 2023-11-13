import torch
from torch import Tensor
from torch import nn
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

import torch.nn.functional as F


def si_sdr(pred, target):

    alpha = (target * pred).sum() / torch.linalg.norm(target)**2
    return 20 * torch.log10(torch.linalg.norm(alpha * target) / (torch.linalg.norm(alpha * target - pred) + 1e-6) + 1e-6)


class SI_SDRLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(SI_SDRLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, s1: Tensor, s2: Tensor, s3: Tensor, audio_target: Tensor, **batch) -> Tensor:
        return - ((1 - self.alpha - self.beta) * si_sdr(pred=s1, target=audio_target) + self.alpha * si_sdr(pred=s2, target=audio_target) + self.beta * si_sdr(pred=s3, target=audio_target))
    

class SpExLoss(nn.Module):
    def __init__(self, alpha, beta, gamma, *args, **kwargs): 
        super(SpExLoss, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.si_sdr = ScaleInvariantSignalDistortionRatio().to(torch.cuda.current_device())

    def forward(self, s1: Tensor, s2: Tensor, s3: Tensor, audio_target: Tensor, logits: Tensor, speaker_id: Tensor, **batch) -> Tensor:
        # si_sdr_loss = super().forward(s1, s2, s3, audio_target, **batch)
        si_sdr_loss = - ((1 - self.alpha - self.beta) * self.si_sdr(s1, audio_target) + self.alpha * self.si_sdr(s2, audio_target) + self.beta * self.si_sdr(s3, audio_target))
        return si_sdr_loss + self.gamma * F.cross_entropy(logits, speaker_id)

    



