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

class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)."""

    def __init__(self, channel_size):
        super().__init__()
        # shape (1, C, 1)
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """Forward.
        Args:
            y: (B, C, L) B - batch size, C - channel size, L - length
        """

        mean = y.mean(dim=(1, 2), keepdim=True)  # (B, 1, 1)
        var = ((y - mean) ** 2).mean(dim=(1, 2), keepdim=True)
        gLN = self.gamma * (y - mean) / (var + 1e-6) ** (1/2) + self.beta

        return gLN


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
    

class TCNBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, speaker_emb_dim, kernel_size, is_first, dilation=1):

        super(TCNBlock, self).__init__()

        padding = dilation * (kernel_size - 1) // 2

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels + speaker_emb_dim if is_first else in_channels, out_channels=hidden_channels, kernel_size=1),
            nn.PReLU(),
            GlobalLayerNorm(channel_size=hidden_channels),
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                groups=hidden_channels
            ),
            nn.PReLU(),
            GlobalLayerNorm(channel_size=hidden_channels),
            nn.Conv1d(in_channels=hidden_channels, out_channels=in_channels, kernel_size=1),
        )


    def forward(self, x, speaker_emb=None):
        if speaker_emb is not None:
            # stack on channels dim
            inp = torch.cat([x, speaker_emb], dim=1)
        else:
            inp = x
        out = x + self.net(inp)
        return out
    
class StackedTCNs(nn.Module):

    def __init__(self, in_channels, hidden_channels, speaker_emb_dim, num_blocks):

        super(StackedTCNs, self).__init__()

        self.first_block = TCNBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            speaker_emb_dim=speaker_emb_dim,
            kernel_size=3,
            is_first=True,
            dilation=1
        )

        tcn_blocks = []
        for i in range(1, num_blocks):
            tcn_blocks.append(
                TCNBlock(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    speaker_emb_dim=speaker_emb_dim,
                    kernel_size=3,
                    is_first=False,
                    dilation=2**i
                )
            )

        self.other_blocks = nn.Sequential(*tcn_blocks)

    def forward(self, x, speaker_emb):
        x = self.first_block(x, speaker_emb)
        x = self.other_blocks(x)
        return x
    
class SpeakerExtractor(nn.Module):

    def __init__(self, N_filters, in_channels, hidden_channels, speaker_emb_dim, num_blocks, num_stacked):
        super(SpeakerExtractor, self).__init__()

        self.num_stacked = num_stacked

        self.speaker_extractor_prep = nn.Sequential(         
            ChannelWiseLayerNorm(normalized_shape=3*N_filters),
            nn.Conv1d(in_channels=3*N_filters, out_channels=in_channels, kernel_size=1)
        )

        self.tcn_stacked_blocks = nn.ModuleList(
            [StackedTCNs(in_channels=in_channels, hidden_channels=hidden_channels, speaker_emb_dim=speaker_emb_dim, num_blocks=num_blocks) for _ in range(num_stacked)]
        )


        self.masks_conv = nn.ModuleList(
            [nn.Conv1d(in_channels=in_channels, out_channels=N_filters, kernel_size=1) for _ in range(3)]
        )

    def forward(self, y, y1, y2, y3, speaker_emb):
        L = y.shape[-1]
        speaker_emb = torch.unsqueeze(speaker_emb, -1).expand(-1, -1, L)

        out = self.speaker_extractor_prep(y)

        for i in range(self.num_stacked):
            out = self.tcn_stacked_blocks[i](out, speaker_emb)

        m1 = F.relu(self.masks_conv[0](out))
        m2 = F.relu(self.masks_conv[1](out))
        m3 = F.relu(self.masks_conv[2](out))

        s1 = y1 * m1
        s2 = y2 * m2
        s3 = y3 * m3
        return s1, s2, s3

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
        # self.L1 = 20
        # self.L2 = 80
        # self.L3 = 160

        # Speech Encoder
        self.N_filters = 256
        self.encoder_short = nn.Conv1d(in_channels=1, out_channels=self.N_filters, kernel_size=self.L1, stride=self.L1 // 2, bias=False)
        self.encoder_middle = nn.Conv1d(in_channels=1, out_channels=self.N_filters, kernel_size=self.L2, stride=self.L1 // 2, bias=False)
        self.encoder_long = nn.Conv1d(in_channels=1, out_channels=self.N_filters, kernel_size=self.L3, stride=self.L1 // 2, bias=False)

        # Speaker Encoder
        self.speaker_encoder = nn.Sequential(
            ChannelWiseLayerNorm(normalized_shape=3*self.N_filters),
            nn.Conv1d(in_channels=3*self.N_filters, out_channels=256, kernel_size=1),
            nn.Sequential(*[ResNetBlock(in_dim=256, hidden_dim=512) for _ in range(N_resnet_blocks)]),
            nn.Conv1d(in_channels=256, out_channels=self.speaker_emb_dim, kernel_size=1)
        )

        self.speaker_encoder_head = nn.Linear(in_features=self.speaker_emb_dim, out_features=self.n_speakers)

        # Speaker Extractor
        
        self.speaker_extractor = SpeakerExtractor(
            N_filters=self.N_filters,
            in_channels=256,
            hidden_channels=512,
            speaker_emb_dim=self.speaker_emb_dim,
            num_blocks=4,
            num_stacked=8
        )

        # Speech Decoder
        self.decoder_short = nn.ConvTranspose1d(in_channels=self.N_filters, out_channels=1, kernel_size=self.L1, stride=self.L1 // 2, bias=False)
        self.decoder_middle = nn.ConvTranspose1d(in_channels=self.N_filters, out_channels=1, kernel_size=self.L2, stride=self.L1 // 2, bias=False)
        self.decoder_long = nn.ConvTranspose1d(in_channels=self.N_filters, out_channels=1, kernel_size=self.L3, stride=self.L1 // 2, bias=False)


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
        y = torch.cat([y1, y2, y3], dim=1)

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
        x = torch.cat([x1, x2, x3], dim=1)

        # Speaker Encoder

        x = self.speaker_encoder(x)
        speaker_emb = x.mean(dim=-1)
        cls_logits = self.speaker_encoder_head(speaker_emb)

        # Speaker Extractor

        s1, s2, s3 = self.speaker_extractor(y, y1, y2, y3, speaker_emb)

        # Speech Decoder

        s1 = self.decoder_short(s1)[:, :, :mix_len]
        s2 = self.decoder_middle(s2)[:, :, :mix_len]
        s3 = self.decoder_long(s3)[:, :, :mix_len]


        return {"s1" : s1, "s2" : s2, "s3" : s3, "logits": cls_logits}


    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
    

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
