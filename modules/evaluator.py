import os

import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt

from fvcore.nn import FlopCountAnalysis, flop_count_table, ActivationCountAnalysis


class Evaluator(object):
    def __init__(self, model, metric_ftns, test_dataloader, args):
        self.args = args
        self.metric_ftns = metric_ftns
        self.test_dataloader = test_dataloader
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        self.reports_path = args.reports_path
        self.output_att_map = args.output_att_map
        self.att_path = args.att_path

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

    def evaluate(self):
        start_time = time.time()
        self.model.eval()
        total_flops = 0
        with torch.no_grad():
            test_gts, test_res = [], []
            with tqdm(desc='Evaluation', unit='it', total=len(self.test_dataloader.dataset)) as pbar:
                for batch_idx, (images_id, images, reports_ids, reports_masks, img_padding_mask) in enumerate(self.test_dataloader):
                    images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                        self.device), reports_masks.to(self.device)
                    if img_padding_mask is not None:
                        img_padding_mask = img_padding_mask.to(self.device)

                    # flops = ActivationCountAnalysis(self.model, (images, images,'sample'))
                    # flops = flops.total()
                    # print(flops.total())
                    # print(flop_count_table(flops))
                    # total_flops += float(flops.total())

                    output, alphas = self.model(images, mode='sample', img_mask=img_padding_mask)
                    reports = self.model.tokenizer.decode_batch(output)
                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                    if self.output_att_map:
                        assert images.shape[0] == 1, 'Only support batch size 1'
                        heatmaps = self.generate_heatmaps(images, alphas)
                        # list (seq_len, num_img, h, w)

                        self.save_heatmaps(heatmaps, output, images_id, self.model.tokenizer.idx2token)

                    test_res.extend(reports)
                    test_gts.extend(ground_truths)

                    pbar.update()

                test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                            {i: [re] for i, re in enumerate(test_res)})
                log = {'test_' + k: v for k, v in test_met.items()}
            with open(self.reports_path, 'w') as f:
                for report in test_res:
                    f.write(report + '\n')
            
            with open('reports/gt.csv', 'w') as f:
                for report in test_gts:
                    f.write(report + '\n')

            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))
            
            print('time elpased: ', time.time() - start_time)
            print('reports are written to {}.'.format(self.reports_path))
            print('Avg. Flops:', total_flops/1e9/len(self.test_dataloader.dataset))

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except RuntimeError:
            for k, v in list(checkpoint['state_dict'].items()):
                new_k = k[7:]
                checkpoint['state_dict'][new_k] = v
                del checkpoint['state_dict'][k]
            self.model.load_state_dict(checkpoint['state_dict'])
        print("Checkpoint loaded.")

    def generate_heatmaps(self, images, alphas):
        assert images.shape[0] == 1

        images = images.squeeze(0)  # (num_img, 3, H, W)
        alphas = alphas.squeeze(0)  # (seq_len, num_img * num_patch)

        seq_len = alphas.shape[0]
        num_img = images.shape[0]
        num_patch = alphas.shape[1] // num_img
        att_len = int(np.sqrt(num_patch))

        alphas = alphas.view(1, seq_len, num_img, num_patch).contiguous()  # (1, seq_len, num_img, num_patch)
        alphas -= alphas - alphas.min(-1, keepdim=True)[0]
        alphas /= alphas.max(-1, keepdim=True)[0] + 1e-8

        # print(alphas.max(-1))

        assert att_len * att_len == num_patch

        alphas = alphas.view(seq_len, num_img, att_len, att_len).contiguous()
        upscaled = F.interpolate(alphas, images.shape[-2:], mode='bilinear', align_corners=True)
        # (seq_len, num_img, H, W)

        upscaled = (255 * upscaled).type(torch.uint8).cpu().numpy()

        # recover the normalized images
        images = images.permute(0, 2, 3, 1).contiguous()
        mean = torch.tensor([0.485, 0.456, 0.406]).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(images.device)
        images = images * std + mean
        # images = images.permute(0, 3, 1, 2).contiguous()
        images = (255 * images).type(torch.uint8).cpu().numpy()  # (num_img, H, W, 3)

        heatmaps = []
        for i in range(seq_len):
            heatmaps_tmp = []
            for j in range(num_img):
                heatmap = upscaled[i, j]
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)[:, :, ::-1]
                heatmap = (0.2 * heatmap + 0.8 * images[j]).astype(np.uint8)
                heatmaps_tmp.append(heatmap)
            heatmaps.append(heatmaps_tmp)

        return heatmaps

    def save_heatmaps(self, heatmaps, reports, images_id, idx2token):
        """
        save each heatmap to self.att_path/image_id/reports[i].png

        :param heatmaps: list of list of heatmaps with shape (seq_len, num_img, H, W, 3)
        :param reports: list of reports with shape (1, seq_len)
        :param images_id: list of image id
        """
        assert len(images_id) == 1
        assert len(reports) == 1

        # print(len(heatmaps), len(reports[0]))

        save_path = os.path.join(self.att_path, str(images_id[0]))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i in range(len(heatmaps)):
            token_idx = int(reports[0][i])
            if token_idx == 0:
                break
            for j in range(len(heatmaps[i])):
                heatmap = heatmaps[i][j]
                plt.imsave(os.path.join(save_path, '{}_{}.png'.format(idx2token[token_idx], j)), heatmap)
