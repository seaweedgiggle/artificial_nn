import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.img_sz = args.img_size

        if self.split == 'trainval':
            self.examples = self.ann['train'] + self.ann['val']
        else:
            self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            tokenized = tokenizer(self.examples[i]['report'])
            self.examples[i]['ids'] = tokenized['input_ids'].squeeze(0)[:self.max_seq_length]
            self.examples[i]['mask'] = tokenized['attention_mask'].squeeze(0)[:self.max_seq_length]

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        label = example['label']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length, None, label)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['study_id']
        image_path = example['image_path']
        images = torch.zeros((4, 3, self.img_sz, self.img_sz))
        img_padding_mask = torch.zeros(4, dtype=torch.bool)
        for i in range(len(image_path)):
            image = Image.open(os.path.join(self.image_dir, image_path[i])).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            images[i] = image
            img_padding_mask[i] = True
        # images = torch.stack(images, 0)
        report_ids = example['ids']
        report_masks = example['mask']
        label = example['label']
        seq_length = len(report_ids)
        sample = (image_id, images, report_ids, report_masks, seq_length, img_padding_mask, label)
        return sample
