from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from torch import Tensor

class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)  # bs x C x L => bs x L x C
        # Layer norm on last dim
        x = super().forward(x)
        x = torch.transpose(x, 1, 2) # bs x C x L => bs x L x C
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):

        super(ResNetBlock, self).__init__()


        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=1),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.PReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=in_dim, kernel_size=1),
            nn.BatchNorm1d(num_features=in_dim)
        )
        self.act = nn.PReLU()
        self.pool = nn.MaxPool1d(kernel_size=3)


    def forward(self, x):
        x = x + self.net(x)
        x = self.act(x)
        x = self.pool(x)
        return x
    

# class TCNBlock(nn.Module):

#     def __init__(self, in_channels, hidden_channels, kernel_size, dilation=1):

#         super(TCNBlock, self).__init__()

#         self.net = nn.Sequential(
#             nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1),
#             nn.LayerNorm([hidden_channels, ]),
#             nn.PReLU(),
#             nn.Conv1d(
#                 in_channels=hidden_channels,
#                 out_channels=hidden_channels,
#                 kernel_size=kernel_size,
#                 dilation=dilation,
#                 groups=hidden_channels
#             ),
#             nn.LayerNorm([]),
#             nn.PReLU(),
#             nn.Conv1d(in_channels=hidden_channels, out_channels=in_channels, kernel_size=1),
#         )


#     def forward(self, x, speaker_emb=None):
#         #print(x.shape)
#         T = x.shape[-1]
#         #print(aux.shape)
#         aux = torch.unsqueeze(aux, -1)
#         #print(aux.shape)
#         aux = aux.repeat(1,1,T)

#         if speaker_emb is not None:
#             x = torch.concat([x, speaker_emb], dim=)
#         self.net
 
#         return x
    
# class StackedTCNs(nn.Module):

#     def __init__(self, in_channels, hidden_channels, num_blocks):

#         super(StackedTCNs, self).__init__()
#         self.first_block = TCNBlock(
#             in_channels=in_channels,
#             hidden_channels=hidden_channels,
#             kernel_size=3
#         )

#         for i in range(1, num_blocks):
#             dilation = 2**i

#         nn.Sequential(

#     def forward(self, x, speaker_emb):

 
#         return x




class SpExPlus(nn.Module):
    def __init__(self, n_speakers, speaker_emb_dim=256, **batch):
        super(SpExPlus, self).__init__()

        self.n_speakers = n_speakers
        self.speaker_emb_dim = speaker_emb_dim

        N_resnet_blocks = 3

        # filter lengths
        # L1, L2, L3 windows are tuned to cover 20(2.5ms), 80(10ms), 160(20ms) samples in this work
        # for speech of 8kHz sampling rate
        # we have 16kHz 
        self.L1 = 40
        self.L2 = 160
        self.L3 = 320

        # Speech Encoder
        self.N_filters = 256
        self.encoder_short = nn.Conv1d(in_channels=1, out_channels=self.N_filters, kernel_size=self.L1, stride=self.L1 // 2)
        self.encoder_middle = nn.Conv1d(in_channels=1, out_channels=self.N_filters, kernel_size=self.L2, stride=self.L1 // 2)
        self.encoder_long = nn.Conv1d(in_channels=1, out_channels=self.N_filters, kernel_size=self.L3, stride=self.L1 // 2)

        # Speaker Encoder
        self.speaker_encoder = nn.Sequential(
            ChannelWiseLayerNorm(normalized_shape=3*self.N_filters),
            nn.Conv1d(in_channels=3*self.N_filters, out_channels=256, kernel_size=1),
            nn.Sequential(*[ResNetBlock(in_dim=256, hidden_dim=512) for _ in range(N_resnet_blocks)]),
            nn.Conv1d(in_channels=256, out_channels=self.speaker_emb_dim, kernel_size=1)
        )
        self.speaker_encoder_head = nn.Linear(in_features=self.speaker_emb_dim, out_features=self.n_speakers)



        # Speaker Extractor
        # self.speaker_extractor = nn.Sequential(
        #     ChannelWiseLayerNorm(normalized_shape=3*self.N_filters),
        #     nn.Conv1d(in_channels=3*self.N_filters, out_channels=speaker_emb_dim, kernel_size=1),
        #     TCNBlock(in_channels=speaker_emb_dim, )

        # )

    def forward(self, audio_mix : Tensor, audio_ref : Tensor, audio_ref_length : Tensor, *args, **kwargs):
        # Speech Encoder
        # Mixture input
        mix_len = audio_mix.shape[-1]
        y1 = F.relu(self.encoder_short(audio_mix))
        L_out = y1.shape[-1]
        
        # adapt L_in to get same L_out for middle and long encoders
        L_in2 = (L_out - 1) * (self.L1 // 2) + self.L2
        L_in3 = (L_out - 1) * (self.L1 // 2) + self.L3
        y2 = F.relu(self.encoder_middle(F.pad(audio_mix, (0, L_in2 - mix_len))))
        y3 = F.relu(self.encoder_long(F.pad(audio_mix, (0, L_in3 - mix_len))))

        # bs x 3N_filters x L_out
        y = torch.concat([y1, y2, y3], dim=1)


        # Reference input

        ref_len = audio_ref.shape[-1]
        x1 = F.relu(self.encoder_short(audio_ref))
        L_out_ref = x1.shape[-1]
        
        # adapt L_in to get same L_out for middle and long encoders
        L_in2_ref = (L_out_ref - 1) * (self.L1 // 2) + self.L2
        L_in3_ref = (L_out_ref - 1) * (self.L1 // 2) + self.L3
        x2 = F.relu(self.encoder_middle(F.pad(audio_ref, (0, L_in2_ref - ref_len))))
        x3 = F.relu(self.encoder_long(F.pad(audio_ref, (0, L_in3_ref - ref_len))))

        # bs x 3N_filters x L_out_ref
        x = torch.concat([x1, x2, x3], dim=1)


        # Speaker Encoder

        x = self.speaker_encoder(x)
        # Mean pooling depends on init ref_lenths + conv changes in Speech Encoder
        # L_out = (L_in - kernel_size) // stride + 1
        new_ref_lenths = (audio_ref_length - self.L1) // (self.L1 // 2) + 1
        speaker_emb = x.sum(dim=-1) / new_ref_lenths.view(-1,1).float()
        cls_logits = self.speaker_encoder_head(speaker_emb)

        # Speaker Extractor

        # speaker_emb

        return {"s1" : None, "s2" : None, "s3" : None, "logits": cls_logits}


    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
    

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
