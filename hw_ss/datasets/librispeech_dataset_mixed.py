import json
import logging
import os
import shutil
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_ss.base.base_dataset import BaseDataset
from hw_ss.mixer import  MixtureGenerator
from hw_ss.utils import ROOT_PATH

from glob import glob

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class LibrispeechDatasetMixed(BaseDataset):
    def __init__(self, part, nfiles, data_dir=None, *args, **kwargs):
        assert part in URL_LINKS or part == 'train_all'

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir

        if part == 'train_all':
            index = sum([self._get_or_load_index(part)
                         for part in URL_LINKS if 'train' in part], [])
        else:
            index = self._get_or_load_index(part)

        self.nfiles = nfiles

        super().__init__(index, *args, **kwargs)


    def _create_mix(self, part, split_dir, out_dir):
        mixer = MixtureGenerator(
            speakers_files_path=split_dir,
            out_folder=out_dir,
            nfiles=self.nfiles,
            test=('test' in part)
        )

        mixer.generate_mixes(
            snr_levels=[0] if 'test' in part else [0, 3, 5],
            num_workers=8,
            update_steps=100,
            trim_db=None if 'test' in part else 20,
            vad_db=20,
            audioLen=3
        )


    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        logger.info(f"Loading part {part}")
        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        out_dir = self._data_dir / f"{part}_mixed"
        if not out_dir.exists():
            self._create_mix(part, split_dir, out_dir)


        ref = sorted(glob(os.path.join(out_dir, "ref", '*-ref.wav')))
        mix = sorted(glob(os.path.join(out_dir, "mix", '*-mixed.wav')))
        target = sorted(glob(os.path.join(out_dir, "target", '*-target.wav')))

        speaker_id_class = -1
        speaker_id_prev = -1
        for i in range(len(ref)):
            speaker_id = ref[i].split('/')[-1].split('_')[0]
            if speaker_id != speaker_id_prev:
                speaker_id_class += 1
            speaker_id_prev = speaker_id
            index.append(
                {
                    "path_ref": ref[i],
                    "path_mix": mix[i],
                    "path_target": target[i],
                    "speaker_id": speaker_id_class
                }
            )
        return index
