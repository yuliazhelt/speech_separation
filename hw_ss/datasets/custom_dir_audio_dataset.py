import logging
from pathlib import Path

from hw_ss.datasets.custom_audio_dataset import CustomAudioDataset
from hw_ss.base.base_dataset import BaseDataset

from glob import glob
import os
logger = logging.getLogger(__name__)


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, mix_dir, ref_dir, target_dir=None, *args, **kwargs):
        index = []

        ref = sorted(glob(os.path.join(ref_dir, '*-ref.wav')))
        mix = sorted(glob(os.path.join(mix_dir, '*-mixed.wav')))
        target = sorted(glob(os.path.join(target_dir, '*-target.wav')))

        for i in range(len(ref)):
            index.append(
                {
                    "path_ref": ref[i],
                    "path_mix": mix[i],
                    "path_target": target[i],
                }
            )
        print(ref_dir)
        super().__init__(index, *args, **kwargs)
