import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, label, targets=None, mode='train', img_mask=None):
        att_feats_0, fc_feats_0,_ = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, _ = self.visual_extractor(images[:, 1]) #(16, 2048) -> (16, 4096)
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        fc_feats_cp = torch.rand_like(fc_feats)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        # self.encoder_decoder 的返回值变成了 decoder 的输出 + 分类器的输出 (两个输出都经过了 softmax )
        # label 已经作为参数传进来了 (bs, 20)
        # 这里等你改了
        if mode == 'train':
            output, classify_output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, None

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats, _ = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, None

