from torch import Tensor
import torch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from hw_ss.base.base_metric import BaseMetric


class SI_SDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = ScaleInvariantSignalDistortionRatio().to(torch.cuda.current_device())

    def __call__(self, s1: Tensor, audio_target: Tensor, **kwargs):
        return self.si_sdr(preds=s1, target=audio_target)

