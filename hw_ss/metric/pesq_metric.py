from torch import Tensor
import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from hw_ss.base.base_metric import BaseMetric


class PESQMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb', n_processes=8).to(torch.cuda.current_device())


    def __call__(self, s1: Tensor, audio_target: Tensor, **kwargs):
        return self.pesq(preds=s1, target=audio_target)
