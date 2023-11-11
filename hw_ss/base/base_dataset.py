import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from hw_ss.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            config_parser: ConfigParser,
            wave_augs=None,
            spec_augs=None,
            limit=None
    ):
        self.config_parser = config_parser
        self.wave_augs = wave_augs
        self.spec_augs = spec_augs
        self.log_spec = config_parser["preprocessing"]["log_spec"]

        self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(index, limit)
        self._index: List[dict] = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]

        audio_ref_path = data_dict["path_ref"]
        audio_ref_wave = self.load_audio(audio_ref_path)
        audio_ref_wave, audio_ref_spec = self.process_wave(audio_ref_wave)

        audio_mix_path = data_dict["path_mix"]
        audio_mix_wave = self.load_audio(audio_mix_path)
        audio_mix_wave, audio_mix_spec = self.process_wave(audio_mix_wave)

        audio_target_path = data_dict["path_target"]
        audio_target_wave = self.load_audio(audio_target_path)
        audio_target_wave, audio_target_spec = self.process_wave(audio_target_wave)
        return {
            "audio_ref": audio_ref_wave,
            "spectrogram_ref": audio_ref_spec,
            "audio_mix": audio_mix_wave,
            "spectrogram_mix": audio_mix_spec,
            "audio_target": audio_target_wave,
            "spectrogram_target": audio_target_spec,
            "duration_mix": audio_mix_wave.size(1) / self.config_parser["preprocessing"]["sr"],
            "duration_ref": audio_ref_wave.size(1) / self.config_parser["preprocessing"]["sr"],
            "path_ref": audio_ref_path,
            "path_mix": audio_mix_path,
            "path_target": audio_target_path,
            "speaker_id": data_dict["speaker_id"]
        }

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def process_wave(self, audio_tensor_wave: Tensor):
        with torch.no_grad():
            if self.wave_augs is not None:
                audio_tensor_wave = self.wave_augs(audio_tensor_wave)
            wave2spec = self.config_parser.init_obj(
                self.config_parser["preprocessing"]["spectrogram"],
                torchaudio.transforms,
            )
            audio_tensor_spec = wave2spec(audio_tensor_wave)
            if self.spec_augs is not None:
                audio_tensor_spec = self.spec_augs(audio_tensor_spec)
            if self.log_spec:
                audio_tensor_spec = torch.log(audio_tensor_spec + 1e-5)
            return audio_tensor_wave, audio_tensor_spec

    @staticmethod
    def _filter_records_from_dataset(
            index: list, limit
    ) -> list:

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "path_ref" in entry, (
                "Each dataset item should include field 'path_ref'" " - path to reference audio file."
            )
            assert "path_mix" in entry, (
                "Each dataset item should include field 'path_mix'" " - path to mixed audio file."
            )
            assert "path_target" in entry, (
                "Each dataset item should include field 'path_target'" " - path to target audio file."
            )
