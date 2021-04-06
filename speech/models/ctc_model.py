from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
from torch.nn import CTCLoss
from . import model
from .ctc_decoder import decode


class CTC(model.Model):
    def __init__(self, freq_dim, output_dim, config):
        super().__init__(freq_dim, config)

        # include the blank token
        self.blank = output_dim
        self.fc = model.LinearND(self.encoder_dim, output_dim + 1)
        self.loss_func = CTCLoss(blank=self.blank)

    def forward(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        return self.forward_impl(x)

    def forward_impl(self, x, softmax=True):
        if self.is_cuda:
            x = x.cuda()
        x = self.encode(x)
        x = self.fc(x)
        if softmax:
            return torch.nn.LogSoftmax(dim=2)(x)
        return x

    def loss(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        out = self.forward_impl(x).permute(1, 0, 2)

        loss = self.loss_func(out, y, x_lens, y_lens)
        return loss

    def collate(self, inputs, labels):
        max_t = max(i.shape[0] for i in inputs)
        max_t = self.conv_out_size(max_t, 0)
        x_lens = torch.full((len(inputs), 1), fill_value=max_t, dtype=torch.long)
        x = model.zero_pad_concat(inputs, 'feat')
        y_lens = torch.tensor([len(l) for l in labels], dtype=torch.long)
        y = model.zero_pad_concat([torch.tensor(l) for l in labels], 'target', fill_value=self.blank)
        batch = [x, y, x_lens, y_lens]
        if self.volatile:
            for v in batch:
                v.volatile = True
        return batch

    def infer_batch(self, batch, calculate_loss=False):
        x, y, x_lens, y_lens = self.collate(*batch)
        probs = self.forward_impl(x, softmax=True).permute(1, 0, 2)
        if calculate_loss:
            loss = self.loss_func(probs, y, x_lens, y_lens)
        else:
            loss = None

        probs = probs.permute(1, 0, 2).data.cpu().numpy()
        return [decode(p, beam_size=10, blank=self.blank)[0] for p in probs],\
               [y[i, :s].tolist() for i, s in enumerate(y_lens)], loss

    def infer_recording(self, rec):
        probs = self.forward_impl(rec, softmax=True)
        probs = probs.data.cpu().numpy()

        return [decode(p, beam_size=10, blank=self.blank)[0] for p in probs]


    @staticmethod
    def max_decode(pred, blank):
        prev = pred[0]
        seq = [prev] if prev != blank else []
        for p in pred[1:]:
            if p != blank and p != prev:
                seq.append(p)
            prev = p
        return seq
