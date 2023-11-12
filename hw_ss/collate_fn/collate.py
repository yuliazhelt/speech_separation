import logging
from typing import List
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence
import random
logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch_list = defaultdict(list)
    result_batch = {}
    
    for item in dataset_items:

        result_batch_list['audio_ref'].append(item['audio_ref'].squeeze(0))
        result_batch_list['audio_ref_length'].append(item['audio_ref'].shape[1])
        # result_batch_list['spectrogram_ref'].append(item['spectrogram_ref'].squeeze(0).T)

        result_batch_list['audio_mix'].append(item['audio_mix'].squeeze(0))
        result_batch_list['audio_mix_length'].append(item['audio_mix'].shape[1])
        # result_batch_list['spectrogram_mix'].append(item['spectrogram_mix'].squeeze(0).T)

        result_batch_list['audio_target'].append(item['audio_target'].squeeze(0))
        result_batch_list['audio_target_length'].append(item['audio_target'].shape[1])
        # result_batch_list['spectrogram_target'].append(item['spectrogram_target'].squeeze(0).T)

        result_batch_list['path_ref'].append(item['path_ref'])
        result_batch_list['path_mix'].append(item['path_mix'])
        result_batch_list['path_target'].append(item['path_target'])

        result_batch_list['speaker_id'].append(item['speaker_id'])


    result_batch['audio_ref'] = pad_sequence(result_batch_list['audio_ref'], batch_first=True)
    result_batch['audio_ref_length'] = torch.tensor(result_batch_list['audio_ref_length'])
    # result_batch['spectrogram_ref'] = pad_sequence(result_batch_list['spectrogram_ref'], batch_first=True).transpose(1, 2)

    result_batch['audio_mix'] = pad_sequence(result_batch_list['audio_mix'], batch_first=True)
    result_batch['audio_mix_length'] = torch.tensor(result_batch_list['audio_mix_length'])
    # result_batch['spectrogram_mix'] = pad_sequence(result_batch_list['spectrogram_mix'], batch_first=True).transpose(1, 2)
    
    result_batch['audio_target'] = pad_sequence(result_batch_list['audio_target'], batch_first=True)
    result_batch['audio_target_length'] = torch.tensor(result_batch_list['audio_target_length'])
    # result_batch['spectrogram_target'] = pad_sequence(result_batch_list['spectrogram_target'], batch_first=True).transpose(1, 2)
    

    result_batch['speaker_id'] = torch.tensor(result_batch_list['speaker_id'])
    
    result_batch['path_ref'] = result_batch_list['path_ref']
    result_batch['path_mix'] = result_batch_list['path_mix']
    result_batch['path_target'] = result_batch_list['path_target']

    result_batch['audio_mix_sample'] = random.choice(dataset_items)['audio_mix'].cpu()

    result_batch['audio_ref'] = result_batch['audio_ref'].unsqueeze(1)
    result_batch['audio_mix'] = result_batch['audio_mix'].unsqueeze(1)
    result_batch['audio_target'] = result_batch['audio_target'].unsqueeze(1)


    return result_batch