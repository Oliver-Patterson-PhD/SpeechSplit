import torch
from hparams import hparams


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, signal):
        return self.conv(signal)


class Encoder_t(torch.nn.Module):
    """Rhythm Encoder"""

    def __init__(self):
        super().__init__()
        self.chs_grp = hparams.chs_grp
        self.dim_emb = hparams.dim_spk_emb
        self.dim_enc_2 = hparams.dim_enc_2
        self.dim_freq = hparams.dim_freq
        self.dim_neck_2 = hparams.dim_neck_2
        self.freq_2 = hparams.freq_2
        convolutions = []
        for i in range(1):
            conv_layer = torch.nn.Sequential(
                ConvNorm(
                    self.dim_freq if i == 0 else self.dim_enc_2,
                    self.dim_enc_2,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="relu",
                ),
                torch.nn.GroupNorm(self.dim_enc_2 // self.chs_grp, self.dim_enc_2),
            )
            convolutions.append(conv_layer)
        self.convolutions = torch.nn.ModuleList(convolutions)
        self.lstm = torch.nn.LSTM(
            self.dim_enc_2, self.dim_neck_2, 1, batch_first=True, bidirectional=True
        )

    def forward(self, x, mask):
        print(f"\t\tx:                          {x.shape}")
        for conv in self.convolutions:
            print("\tloop")
            x = torch.nn.functional.relu(conv(x))
            print(f"\t\tx:                          {x.shape}")
            print("\tendloop")
        x = x.transpose(1, 2)
        print(f"\t\tx:                          {x.shape}")
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        print(f"\t\toutputs:                    {outputs.shape}")
        if mask is not None:
            outputs = outputs * mask
        out_forward = outputs[:, :, : self.dim_neck_2]
        out_backward = outputs[:, :, self.dim_neck_2 :]
        print(f"\t\tout_forward:                {out_forward.shape}")
        print(f"\t\tout_backward:               {out_backward.shape}")
        code_1 = out_forward[:, self.freq_2 - 1 :: self.freq_2, :]
        code_2 = out_backward[:, :: self.freq_2, :]
        print(f"\t\tcode_1:                     {code_1.shape}")
        print(f"\t\tcode_2:                     {code_2.shape}")
        codes = torch.cat((code_1, code_2), dim=-1)
        print(f"\t\tcodes:                      {codes.shape}")
        return codes


class Encoder_6(torch.nn.Module):
    """F0 encoder"""

    def __init__(self):
        super().__init__()
        self.chs_grp = hparams.chs_grp
        self.dim_emb = hparams.dim_spk_emb
        self.dim_enc_3 = hparams.dim_enc_3
        self.dim_f0 = hparams.dim_f0
        self.dim_neck_3 = hparams.dim_neck_3
        self.freq_3 = hparams.freq_3
        self.register_buffer("len_org", torch.tensor(hparams.max_len_pad))
        convolutions = []
        for i in range(3):
            conv_layer = torch.nn.Sequential(
                ConvNorm(
                    self.dim_f0 if i == 0 else self.dim_enc_3,
                    self.dim_enc_3,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="relu",
                ),
                torch.nn.GroupNorm(self.dim_enc_3 // self.chs_grp, self.dim_enc_3),
            )
            convolutions.append(conv_layer)
        self.convolutions = torch.nn.ModuleList(convolutions)
        self.lstm = torch.nn.LSTM(
            self.dim_enc_3, self.dim_neck_3, 1, batch_first=True, bidirectional=True
        )
        self.interp = InterpLnr()

    def forward(self, x):
        for conv in self.convolutions:
            x = torch.nn.relu(conv(x))
            x = x.transpose(1, 2)
            x = self.interp(x, self.len_org.expand(x.size(0)))
            x = x.transpose(1, 2)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, : self.dim_neck_3]
        out_backward = outputs[:, :, self.dim_neck_3 :]
        codes = torch.cat(
            (
                out_forward[:, self.freq_3 - 1 :: self.freq_3, :],
                out_backward[:, :: self.freq_3, :],
            ),
            dim=-1,
        )
        return codes


class Encoder_7(torch.nn.Module):
    """Sync Encoder module"""

    def __init__(self):
        super().__init__()

        self.chs_grp = hparams.chs_grp
        self.dim_enc = hparams.dim_enc
        self.dim_enc_3 = hparams.dim_enc_3
        self.dim_f0 = hparams.dim_f0
        self.dim_freq = hparams.dim_freq
        self.dim_neck = hparams.dim_neck
        self.dim_neck_3 = hparams.dim_neck_3
        self.freq = hparams.freq
        self.freq_3 = hparams.freq_3
        self.register_buffer("len_org", torch.tensor(hparams.max_len_pad))
        # convolutions for code 1
        convolutions = []
        for i in range(3):
            conv_layer = torch.nn.Sequential(
                ConvNorm(
                    self.dim_freq if i == 0 else self.dim_enc,
                    self.dim_enc,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="relu",
                ),
                torch.nn.GroupNorm(self.dim_enc // self.chs_grp, self.dim_enc),
            )
            convolutions.append(conv_layer)
        self.convolutions_1 = torch.nn.ModuleList(convolutions)
        self.lstm_1 = torch.nn.LSTM(
            self.dim_enc, self.dim_neck, 2, batch_first=True, bidirectional=True
        )
        # convolutions for f0
        convolutions = []
        for i in range(3):
            conv_layer = torch.nn.Sequential(
                ConvNorm(
                    self.dim_f0 if i == 0 else self.dim_enc_3,
                    self.dim_enc_3,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="relu",
                ),
                torch.nn.GroupNorm(self.dim_enc_3 // self.chs_grp, self.dim_enc_3),
            )
            convolutions.append(conv_layer)
        self.convolutions_2 = torch.nn.ModuleList(convolutions)
        self.lstm_2 = torch.nn.LSTM(
            self.dim_enc_3, self.dim_neck_3, 1, batch_first=True, bidirectional=True
        )
        self.interp = InterpLnr()

    def forward(self, x_f0):
        print(f"\t\tx_f0:                       {x_f0.shape}")
        x = x_f0[:, : self.dim_freq, :]
        f0 = x_f0[:, self.dim_freq :, :]
        print(f"\t\tx:                          {x.shape}")
        print(f"\t\tf0:                         {f0.shape}")
        for conv_1, conv_2 in zip(self.convolutions_1, self.convolutions_2):
            print("\tloop")
            x = torch.nn.functional.relu(conv_1(x))
            f0 = torch.nn.functional.relu(conv_2(f0))
            print("\t\trelu(conv_x(x,f0))")
            print(f"\t\tx:                          {x.shape}")
            print(f"\t\tf0:                         {f0.shape}")
            x_f0 = torch.cat((x, f0), dim=1)
            print(f"\t\tx_f0:                       {x_f0.shape}")
            x_f0 = x_f0.transpose(1, 2)
            print(f"\t\tx_f0:                       {x_f0.shape}")
            expandlen = self.len_org.expand(x.size(0))
            print(f"\t\texpandlen:                  {expandlen.shape}")
            x_f0 = self.interp(x_f0, expandlen)
            print(f"\t\tx_f0:                       {x_f0.shape}")
            x_f0 = x_f0.transpose(1, 2)
            print(f"\t\tx_f0:                       {x_f0.shape}")
            x = x_f0[:, : self.dim_enc, :]
            f0 = x_f0[:, self.dim_enc :, :]
            print(f"\t\tx:                          {x.shape}")
            print(f"\t\tf0:                         {f0.shape}")
            print("\tendloop")
        print(f"\t\tx_f0:                       {x_f0.shape}")
        x_f0 = x_f0.transpose(1, 2)
        print(f"\t\tx_f0:                       {x_f0.shape}")
        x = x_f0[:, :, : self.dim_enc]
        f0 = x_f0[:, :, self.dim_enc :]
        print(f"\t\tx:                          {x.shape}")
        print(f"\t\tf0:                         {f0.shape}")
        print("\tcode 1")
        # code 1

        x = self.lstm_1(x)[0]
        print(f"\t\tx:                          {x.shape}")
        x_forward = x[:, :, : self.dim_neck]
        x_backward = x[:, :, self.dim_neck :]
        print(f"\t\tx_forward:                  {x_forward.shape}")
        print(f"\t\tx_backward:                 {x_backward.shape}")
        x_part_1 = x_forward[:, self.freq - 1 :: self.freq, :]
        x_part_2 = x_backward[:, :: self.freq, :]
        print(f"\t\tx_part_1:                   {x_part_1.shape}")
        print(f"\t\tx_part_2:                   {x_part_2.shape}")
        codes_x = torch.cat((x_part_1, x_part_2), dim=-1)
        print(f"\t\tcodes_x:                    {codes_x.shape}")

        f0 = self.lstm_2(f0)[0]
        print(f"\t\tf0:                         {f0.shape}")
        f0_forward = f0[:, :, : self.dim_neck_3]
        f0_backward = f0[:, :, self.dim_neck_3 :]
        print(f"\t\tf0_forward:                 {f0_forward.shape}")
        print(f"\t\tf0_backward:                {f0_backward.shape}")
        f0_part_1 = f0_forward[:, self.freq_3 - 1 :: self.freq_3, :]
        f0_part_2 = f0_backward[:, :: self.freq_3, :]
        print(f"\t\tf0_part_1:                  {f0_part_1.shape}")
        print(f"\t\tf0_part_2:                  {f0_part_2.shape}")
        codes_f0 = torch.cat((f0_part_1, f0_part_2), dim=-1)
        print(f"\t\tcodes_f0:                   {codes_f0.shape}")
        return codes_x, codes_f0


class Decoder_3(torch.nn.Module):
    """Decoder module"""

    def __init__(self):
        super().__init__()
        self.dim_emb = hparams.dim_spk_emb
        self.dim_freq = hparams.dim_freq
        self.dim_neck = hparams.dim_neck
        self.dim_neck_2 = hparams.dim_neck_2
        self.dim_neck_3 = hparams.dim_neck_3
        self.linear_projection = LinearNorm(1024, self.dim_freq)
        self.lstm = torch.nn.LSTM(
            self.dim_neck * 2
            + self.dim_neck_2 * 2
            + self.dim_neck_3 * 2
            + self.dim_emb,
            512,
            3,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        outputs, _ = self.lstm(x)
        decoder_output = self.linear_projection(outputs)
        return decoder_output


class Decoder_4(torch.nn.Module):
    """For F0 converter"""

    def __init__(self, hparams):
        super().__init__()
        self.dim_f0 = hparams.dim_f0
        self.dim_neck_2 = hparams.dim_neck_2
        self.dim_neck_3 = hparams.dim_neck_3
        self.linear_projection = LinearNorm(512, self.dim_f0)
        self.lstm = torch.nn.LSTM(
            self.dim_neck_2 * 2 + self.dim_neck_3 * 2,
            256,
            2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        outputs, _ = self.lstm(x)
        decoder_output = self.linear_projection(outputs)
        return decoder_output


class Generator_3(torch.nn.Module):
    """SpeechSplit model"""

    def __init__(self):
        super().__init__()
        self.decoder = Decoder_3()
        self.encoder_1 = Encoder_7()
        self.encoder_2 = Encoder_t()
        self.freq = hparams.freq
        self.freq_2 = hparams.freq_2
        self.freq_3 = hparams.freq_3

    def forward(self, x_f0, x_org, c_trg):
        print(f"\tx_f0:                     {x_f0.shape}")
        print(f"\tx_org:                    {x_org.shape}")
        print(f"\tc_trg:                    {c_trg.shape}")
        x_1 = x_f0.transpose(2, 1)
        x_2 = x_org.transpose(2, 1)
        print(f"\tx_1:                      {x_1.shape}")
        print(f"\tx_2:                      {x_2.shape}")
        assert x_1.shape[2] == 192, f"x_1 shape incorrect: {x_1.shape}"

        print("Sync Encoder:")
        codes_x, codes_f0 = self.encoder_1(x_1)
        print("Rhythm Encoder:")
        codes_2 = self.encoder_2(x_2, None)
        print("SpeechSplit:")
        print(f"\tcodes_x:                  {codes_x.shape}")
        print(f"\tcodes_f0:                 {codes_f0.shape}")
        print(f"\tcodes_2:                  {codes_2.shape}")

        code_exp_1 = codes_x.repeat_interleave(self.freq, dim=1)
        code_exp_2 = codes_2.repeat_interleave(self.freq_2, dim=1)
        code_exp_3 = codes_f0.repeat_interleave(self.freq_3, dim=1)
        print(f"\tcodes_exp_1:              {codes_x.shape}")
        print(f"\tcodes_exp_2:              {codes_f0.shape}")
        print(f"\tcodes_exp_3:              {codes_2.shape}")

        c_step_1 = c_trg.unsqueeze(1)
        c_step_2 = c_step_1.expand(-1, x_1.size(-1), -1)
        print(f"\tc_step_1:                 {c_step_1.shape}")
        print(f"\tc_step_2:                 {c_step_2.shape}")
        encoder_outputs = torch.cat(
            (
                code_exp_1,
                code_exp_2,
                code_exp_3,
                c_step_2,
            ),
            dim=-1,
        )
        print(f"\tencoder_outputs:          {encoder_outputs.shape}")
        mel_outputs = self.decoder(encoder_outputs)
        print(f"\tmel_outputs:              {mel_outputs.shape}")
        return mel_outputs

    def rhythm(self, x_org):
        x_2 = x_org.transpose(2, 1)
        codes_2 = self.encoder_2(x_2, None)
        return codes_2


class Generator_6(torch.nn.Module):
    """F0 converter"""

    def __init__(self, hparams):
        super().__init__()
        self.decoder = Decoder_4(hparams)
        self.encoder_2 = Encoder_t(hparams)
        self.encoder_3 = Encoder_6(hparams)
        self.freq_2 = hparams.freq_2
        self.freq_3 = hparams.freq_3

    def forward(self, x_org, f0_trg):
        x_2 = x_org.transpose(2, 1)
        codes_2 = self.encoder_2(x_2, None)
        code_exp_2 = codes_2.repeat_interleave(self.freq_2, dim=1)
        x_3 = f0_trg.transpose(2, 1)
        codes_3 = self.encoder_3(x_3)
        code_exp_3 = codes_3.repeat_interleave(self.freq_3, dim=1)
        encoder_outputs = torch.cat((code_exp_2, code_exp_3), dim=-1)
        mel_outputs = self.decoder(encoder_outputs)
        return mel_outputs


class InterpLnr(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_len_pad = hparams.max_len_pad
        self.max_len_seg = hparams.max_len_seg
        self.max_len_seq = hparams.max_len_seq
        self.min_len_seg = hparams.min_len_seg
        self.max_num_seg = self.max_len_seq // self.min_len_seg + 1

    def pad_sequences(self, sequences):
        channel_dim = sequences[0].size()[-1]
        out_dims = (len(sequences), self.max_len_pad, channel_dim)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length, :] = tensor[: self.max_len_pad]
        return out_tensor

    def forward(self, x, len_seq):
        if not self.training:
            return x
        device = x.device
        batch_size = x.size(0)
        # indices of each sub segment
        indices = (
            torch.arange(self.max_len_seg * 2, device=device)
            .unsqueeze(0)
            .expand(batch_size * self.max_num_seg, -1)
        )
        # scales of each sub segment
        scales = torch.rand(batch_size * self.max_num_seg, device=device) + 0.5
        idx_scaled = indices / scales.unsqueeze(-1)
        idx_scaled_fl = torch.floor(idx_scaled)
        lambda_ = idx_scaled - idx_scaled_fl
        len_seg = torch.randint(
            low=self.min_len_seg,
            high=self.max_len_seg,
            size=(batch_size * self.max_num_seg, 1),
            device=device,
        )
        # end point of each segment
        idx_mask = idx_scaled_fl < (len_seg - 1)
        offset = len_seg.view(batch_size, -1).cumsum(dim=-1)
        # offset starts from the 2nd segment
        offset = torch.nn.functional.pad(offset[:, :-1], (1, 0), value=0).view(-1, 1)
        idx_scaled_org = idx_scaled_fl + offset
        len_seq_rp = torch.repeat_interleave(len_seq, self.max_num_seg)
        idx_mask_org = idx_scaled_org < (len_seq_rp - 1).unsqueeze(-1)
        idx_mask_final = idx_mask & idx_mask_org
        counts = idx_mask_final.sum(dim=-1).view(batch_size, -1).sum(dim=-1)
        index_1 = torch.repeat_interleave(
            torch.arange(batch_size, device=device), counts
        )
        index_2_fl = idx_scaled_org[idx_mask_final].long()
        index_2_cl = index_2_fl + 1
        y_fl = x[index_1, index_2_fl, :]
        y_cl = x[index_1, index_2_cl, :]
        lambda_f = lambda_[idx_mask_final].unsqueeze(-1)
        y = (1 - lambda_f) * y_fl + lambda_f * y_cl
        sequences = torch.split(y, counts.tolist(), dim=0)
        seq_padded = self.pad_sequences(sequences)
        return seq_padded
