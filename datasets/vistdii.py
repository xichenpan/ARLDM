import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer

from models.blip_override.blip import init_tokenizer


class StoryDataset(Dataset):
    """
    A custom subset class for the LRW (includes train, val, test) subset
    """

    def __init__(self, subset, args):
        super(StoryDataset, self).__init__()
        self.args = args

        self.h5_file = args.get(args.dataset).hdf5_file
        self.subset = subset

        self.augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(512),
            transforms.RandomCrop(512) if self.subset == 'train' else transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]) if self.subset in ['train', 'val'] else transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(64),
            transforms.CenterCrop(64)
        ])

        self.dataset = args.dataset
        self.max_length = args.get(args.dataset).max_length
        self.clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
        self.blip_tokenizer = init_tokenizer()

        self.blip_image_processor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.RandomCrop(224) if self.subset == 'train' else transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])

    def open_h5(self):
        h5 = h5py.File(self.h5_file, "r")
        self.h5 = h5[self.subset]

    def __getitem__(self, index):
        if not hasattr(self, 'h5'):
            self.open_h5()

        images = list()
        for i in range(5):
            im = self.h5['image{}'.format(i)][index]
            im = cv2.imdecode(im, cv2.IMREAD_COLOR)
            images.append(im)

        source_images = torch.stack([self.blip_image_processor(im) for im in images])
        images = images[1:] if self.args.task == 'continuation' else images
        images = [self.augment(im) for im in images]
        images = torch.stack(images) if self.subset in ['train', 'val'] \
            else torch.from_numpy(np.array([np.array(im) for im in images]))

        texts = self.h5['dii'][index].decode('utf-8').split('|')

        # tokenize caption using default tokenizer
        tokenized = self.clip_tokenizer(
            texts[1:] if self.args.task == 'continuation' else texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        captions, attention_mask = tokenized['input_ids'], tokenized['attention_mask']

        tokenized = self.blip_tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        source_caption, source_attention_mask = tokenized['input_ids'], tokenized['attention_mask']
        return images, captions, attention_mask, source_images, source_caption, source_attention_mask

    def __len__(self):
        if not hasattr(self, 'h5'):
            self.open_h5()
        return len(self.h5['dii'])
