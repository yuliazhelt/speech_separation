from torch import Tensor

from hw_ss.base.base_metric import BaseMetric


class SpeakerAccuracyMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, logits: Tensor, speaker_id: Tensor, **kwargs):
        return (logits.argmax(dim=1) == speaker_id).float().mean()
        
