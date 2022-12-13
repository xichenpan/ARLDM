import inspect

import KIT  # This is an internal toolkit
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, LMSDiscreteScheduler, PNDMScheduler, DDIMScheduler
from omegaconf import DictConfig
from torch import nn
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel

from models.blip.blip import blip_feature_extractor, init_tokenizer
from models.diffusers.unet_2d_condition import UNet2DConditionModel
from models.inception import InceptionV3


class ARLDM(nn.Module):
    def __init__(self, args: DictConfig):
        super(ARLDM, self).__init__()
        self.args = args
        """
            Configurations
        """
        self.task = args.task

        if args.mode == 'sample':
            self.num_inference_steps = args.num_inference_steps
            if args.scheduler == "pndm":
                self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                               skip_prk_steps=True, tensor_format="pt")
            elif args.scheduler == "ddim":
                self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                               clip_sample=False, set_alpha_to_one=True, tensor_format="pt")
            else:
                raise ValueError("Scheduler not supported")
            self.fid_augment = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.inception = InceptionV3([block_idx])

        # Data Augmentation
        self.augment = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.dataset = args.dataset
        self.max_length = args.get(args.dataset).max_length
        self.clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
        self.blip_tokenizer = init_tokenizer()
        if self.dataset in ['flintstones', 'pororo']:
            msg = self.clip_tokenizer.add_tokens(list(args.new_tokens))
            print("clip {} new tokens added".format(msg))
            msg = self.blip_tokenizer.add_tokens(list(args.new_tokens))
            print("blip {} new tokens added".format(msg))

        self.blip_image_processor = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        self.square_crop = transforms.CenterCrop(512) if self.sample else transforms.RandomCrop(512)

        blip_image_null_token = self.blip_image_processor(
            Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))).unsqueeze(0).float()
        clip_text_null_token = self.clip_tokenizer([""], padding="max_length", max_length=self.max_length,
                                                   return_tensors="pt").input_ids
        blip_text_null_token = self.blip_tokenizer([""], padding="max_length", max_length=self.max_length,
                                                   return_tensors="pt").input_ids

        self.register_buffer('clip_text_null_token', clip_text_null_token)
        self.register_buffer('blip_text_null_token', blip_text_null_token)
        self.register_buffer('blip_image_null_token', blip_image_null_token)

        self.tokenizer = self.clip_tokenizer
        self.image_processor = self.blip_image_processor

        self.text_encoder = CLIPTextModel.from_pretrained('runwayml/stable-diffusion-v1-5',
                                                          subfolder="text_encoder")
        self.text_encoder.resize_token_embeddings(args.clip_embedding_tokens)
        self.text_encoder.resize_position_embeddings(self.max_length)
        self.modal_type_embeddings = nn.Embedding(2, 768)
        self.time_embeddings = nn.Embedding(5, 768)
        self.mm_encoder = blip_feature_extractor(
            pretrained='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth',
            image_size=224, vit='large')
        self.mm_encoder.text_encoder.resize_token_embeddings(args.blip_embedding_tokens)

        self.vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="unet")
        self.noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                             num_train_timesteps=1000, tensor_format="pt")

        # Freeze vae and unet
        self.freeze_params(self.vae.parameters())
        if args.freeze_resnet:
            self.freeze_params([p for n, p in self.unet.named_parameters() if "attentions" not in n])

        if args.freeze_blip and hasattr(self, "mm_encoder"):
            self.freeze_params(self.mm_encoder.parameters())
            self.unfreeze_params(self.mm_encoder.text_encoder.embeddings.word_embeddings.parameters())

        if self.freeze_clip_text and hasattr(self, "text_encoder"):
            self.freeze_params(self.text_encoder.parameters())
            self.unfreeze_params(self.text_encoder.text_model.embeddings.token_embedding.parameters())

    @staticmethod
    def freeze_params(params):
        for param in params:
            param.requires_grad = False

    @staticmethod
    def unfreeze_params(params):
        for param in params:
            param.requires_grad = True

    def data_preprocess(self, features):
        features['source_image'], features['images'], features['captions'], features['attention_mask'], features[
            'source_caption'], features['source_attention_mask'] = None, None, None, None, None, None
        if self.dataset == "flintstones":
            indices = np.random.randint(5, size=5) + np.arange(0, 25, 5)
            images = np.array(np.split(features['img'], range(128, len(features['img'][0]), 128), axis=1))[indices]
        elif self.dataset == "pororo":
            images = list()
            for i in range(5):
                im = features['img{}'.format(i + 1)]
                idx = np.random.randint(im.shape[0] / 128)
                images.append(im[idx * 128: (idx + 1) * 128, 0: 128])
            images = np.array(images)
        elif self.dataset == "vist":
            images = list()
            for i in range(5):
                im = features['img{}'.format(i + 1)]
                im = Image.fromarray(im.astype('uint8'), 'RGB')
                im = self.square_crop(im)
                images.append(np.array(im))
            images = np.array(images)
        else:
            raise NotImplementedError
        features['img'] = torch.from_numpy(images)
        images = images.astype('uint8')
        images = [Image.fromarray(im, 'RGB') for im in images]
        features['images'] = torch.stack([self.augment(im) for im in images[1:]]) if self.task == 'continuation' \
            else torch.stack([self.augment(im) for im in images])

        # source image processing using default image_processor
        features['source_image'] = torch.stack([self.image_processor(im) for im in images])

        if self.dataset == 'vist':
            texts = features[self.args.caption].split('|')
            texts = [t.strip().strip('\"').strip('\'').strip() for t in texts]
            texts = sorted(set(texts), key=texts.index)
        else:
            texts = features['text'].split('|')
            texts = [t.strip().strip('\"').strip('\'').strip() for t in texts]
        # tokenize caption using default tokenizer
        tokenized = self.tokenizer(
            texts[1:] if self.task == 'continuation' else texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        features['captions'], features['attention_mask'] = tokenized['input_ids'], tokenized['attention_mask']

        tokenized = self.blip_tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        features['source_caption'], features['source_attention_mask'] = \
            tokenized['input_ids'], tokenized['attention_mask']

    def forward(self, input_data, kit_output=None, cur_step=None, guidance_scale=7.5):
        if self.sample:
            return self.do_sampling(input_data, kit_output, guidance_scale)
        if self.freeze_clip_text and hasattr(self, "text_encoder"):
            self.text_encoder.eval()
        if self.freeze_clip_image and hasattr(self, "image_encoder"):
            self.image_encoder.eval()
        if self.freeze_blip and hasattr(self, "mm_encoder"):
            self.mm_encoder.eval()

        source_image, images, captions, attention_mask = \
            input_data.source_image, input_data.images, input_data.captions, input_data.attention_mask
        device = captions.device
        B, V, S = captions.shape
        src_V = V + 1 if self.task == 'continuation' else V

        captions = torch.flatten(captions, 0, 1)
        images = torch.flatten(images, 0, 1)
        attention_mask = torch.flatten(attention_mask, 0, 1)
        # 1 is not masked, 0 is masked

        classifier_free_idx = np.random.rand(B * V) < 0.1

        source_caption, source_attention_mask = input_data.source_caption, input_data.source_attention_mask
        source_caption = torch.flatten(source_caption, 0, 1)
        source_attention_mask = torch.flatten(source_attention_mask, 0, 1)
        source_image = torch.flatten(source_image, 0, 1)

        caption_embeddings = self.text_encoder(captions, attention_mask).last_hidden_state  # B * V, S, D
        source_embeddings = self.mm_encoder(source_image, source_caption, source_attention_mask,
                                            mode='multimodal').reshape(B, src_V * S, -1)
        source_embeddings = source_embeddings.repeat_interleave(V, dim=0)
        caption_embeddings[classifier_free_idx] = \
            self.text_encoder(self.clip_text_null_token).last_hidden_state[0]
        source_embeddings[classifier_free_idx] = \
            self.mm_encoder(self.blip_image_null_token, self.blip_text_null_token, attention_mask=None,
                            mode='multimodal')[0].repeat(src_V, 1)
        caption_embeddings += self.modal_type_embeddings(torch.tensor(0, device=device))
        source_embeddings += self.modal_type_embeddings(torch.tensor(1, device=device))
        source_embeddings += self.time_embeddings(
            torch.arange(src_V, device=device).repeat_interleave(S, dim=0))
        encoder_hidden_states = torch.cat([caption_embeddings, source_embeddings], dim=1)

        attention_mask = torch.cat(
            [attention_mask, source_attention_mask.reshape(B, src_V * S).repeat_interleave(V, dim=0)], dim=1)
        attention_mask = ~(attention_mask.bool())  # B * V, (src_V + 1) * S
        attention_mask[classifier_free_idx] = False

        # B, V, V, S
        square_mask = torch.triu(torch.ones((V, V), device=device)).bool()
        square_mask = square_mask.unsqueeze(0).unsqueeze(-1).expand(B, V, V, S)
        square_mask = square_mask.reshape(B * V, V * S)
        attention_mask[:, -V * S:] = torch.logical_or(square_mask, attention_mask[:, -V * S:])

        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215

        noise = torch.randn(latents.shape, device=device)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, attention_mask).sample
        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
        kit_output.add_eval_output("Loss", loss)
        return loss

    def do_sampling(self, input_data: KIT.TensorStruct, kit_output=None, guidance_scale=7.5):
        source_image, images, captions, attention_mask = \
            input_data.source_image, input_data.images, input_data.captions, input_data.attention_mask
        device = captions.device
        B, V, S = captions.shape
        src_V = V + 1 if self.task == 'continuation' else V

        captions = torch.flatten(captions, 0, 1)
        attention_mask = torch.flatten(attention_mask, 0, 1)
        # 1 is not masked, 0 is masked

        source_caption, source_attention_mask = input_data.source_caption, input_data.source_attention_mask
        source_caption = torch.flatten(source_caption, 0, 1)
        source_attention_mask = torch.flatten(source_attention_mask, 0, 1)

        source_image = torch.flatten(source_image, 0, 1)
        caption_embeddings = self.text_encoder(captions, attention_mask).last_hidden_state  # B * V, S, D
        source_embeddings = self.mm_encoder(source_image, source_caption, source_attention_mask,
                                            mode='multimodal').reshape(B, src_V * S, -1)
        caption_embeddings += self.modal_type_embeddings(torch.tensor(0, device=device))
        source_embeddings += self.modal_type_embeddings(torch.tensor(1, device=device))
        source_embeddings += self.time_embeddings(
            torch.arange(src_V, device=device).repeat_interleave(S, dim=0))
        source_embeddings = source_embeddings.repeat_interleave(V, dim=0)
        encoder_hidden_states = torch.cat([caption_embeddings, source_embeddings], dim=1)

        attention_mask = torch.cat(
            [attention_mask, source_attention_mask.reshape(B, src_V * S).repeat_interleave(V, dim=0)], dim=1)
        attention_mask = ~(attention_mask.bool())  # B * V, (src_V + 1) * S
        # B, V, V, S
        square_mask = torch.triu(torch.ones((V, V), device=device)).bool()
        square_mask = square_mask.unsqueeze(0).unsqueeze(-1).expand(B, V, V, S)
        square_mask = square_mask.reshape(B * V, V * S)
        attention_mask[:, -V * S:] = torch.logical_or(square_mask, attention_mask[:, -V * S:])

        uncond_caption_embeddings = self.text_encoder(self.clip_text_null_token).last_hidden_state
        uncond_source_embeddings = self.mm_encoder(self.blip_image_null_token, self.blip_text_null_token,
                                                   attention_mask=None, mode='multimodal').repeat(1, src_V, 1)
        uncond_caption_embeddings += self.modal_type_embeddings(torch.tensor(0, device=device))
        uncond_source_embeddings += self.modal_type_embeddings(torch.tensor(1, device=device))
        uncond_source_embeddings += self.time_embeddings(
            torch.arange(src_V, device=device).repeat_interleave(S, dim=0))
        uncond_embeddings = torch.cat([uncond_caption_embeddings, uncond_source_embeddings], dim=1)
        uncond_embeddings = uncond_embeddings.expand(B * V, -1, -1)

        encoder_hidden_states = torch.cat([uncond_embeddings, encoder_hidden_states])
        uncond_attention_mask = torch.zeros((B * V, (src_V + 1) * S), device=device).bool()
        uncond_attention_mask[:, -V * S:] = square_mask
        attention_mask = torch.cat([uncond_attention_mask, attention_mask], dim=0)

        attention_mask = attention_mask.reshape(2, B, V, (src_V + 1) * S)
        images = list()
        for i in range(V):
            encoder_hidden_states = encoder_hidden_states.reshape(2, B, V, (src_V + 1) * S, -1)
            new_image = self.diffusion(encoder_hidden_states[:, :, i].reshape(2 * B, (src_V + 1) * S, -1),
                                       attention_mask[:, :, i].reshape(2 * B, (src_V + 1) * S), device,
                                       512, 512, self.num_inference_steps, guidance_scale, 0.0)
            images += new_image

            new_image = torch.stack([self.blip_image_processor(im) for im in new_image]).to(device)
            new_embedding = self.mm_encoder(new_image,  # B,C,H,W
                                            source_caption.reshape(B, src_V, S)[:, i + src_V - V],
                                            source_attention_mask.reshape(B, src_V, S)[:, i + src_V - V],
                                            mode='multimodal')  # B, S, D
            new_embedding = new_embedding.repeat_interleave(V, dim=0)
            new_embedding += self.modal_type_embeddings(torch.tensor(1, device=device))
            new_embedding += self.time_embeddings(torch.tensor(i + src_V - V, device=device))

            encoder_hidden_states = encoder_hidden_states[1].reshape(B * V, (src_V + 1) * S, -1)
            encoder_hidden_states[:, (i + 1 + src_V - V) * S:(i + 2 + src_V - V) * S] = new_embedding
            encoder_hidden_states = torch.cat([uncond_embeddings, encoder_hidden_states])

        if self.sample_images or self.output_human_eval:
            original_images = input_data.img[:, 1:] if self.task == 'continuation' else input_data.img
            original_images = torch.flatten(original_images.transpose(0, 1), start_dim=0, end_dim=1).float()
            original_images = original_images.permute(0, 3, 1, 2)
            original_images = F.interpolate(original_images, size=(512, 512), mode='bilinear',
                                            align_corners=False).permute(0, 2, 3, 1)
            original_images = original_images.cpu().numpy().round().astype('uint8')

            source_images = input_data.img[:, 0].float()
            source_images = source_images.permute(0, 3, 1, 2)
            source_images = F.interpolate(source_images, size=(512, 512), mode='bilinear',
                                          align_corners=False).permute(0, 2, 3, 1)
            source_images = source_images.cpu().numpy().round().astype('uint8')
            if self.dataset == 'vist':
                kit_output.add_caption_output(input_data.getattr(self.args.caption))
            else:
                kit_output.add_caption_output(input_data.text)
        if self.sample_images:
            kit_output.add_image_output([
                Image.fromarray(np.concatenate([np.array(image), original_images[i]], axis=1).astype(np.uint8), 'RGB')
                for i, image in enumerate(images)])
        if self.output_human_eval:
            kit_output.add_human_eval_output(
                image=images,
                source_image=[Image.fromarray(source_image.astype(np.uint8), 'RGB') for source_image in
                              source_images],
                original_image=[Image.fromarray(original_image.astype(np.uint8), 'RGB') for original_image in
                                original_images]
            )
        if self.sample_features:
            kit_output.add_feature_output(self.inception_feature(images, device))
        if self.calculate_fid:
            original_images = input_data.img[:, 1:] if self.task == 'continuation' else input_data.img
            original_images = torch.flatten(original_images, start_dim=0, end_dim=1)
            original_images = original_images.cpu().numpy().astype('uint8')
            original_images = [Image.fromarray(im, 'RGB') for im in original_images]
            kit_output.add_original_feature_output(self.inception_feature(original_images, device))

    def diffusion(self, encoder_hidden_states, attention_mask, device, height, width, num_inference_steps,
                  guidance_scale, eta):
        latents = torch.randn((encoder_hidden_states.shape[0] // 2, self.unet.in_channels, height // 8, width // 8),
                              device=device)

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)

            # noise_pred = self.unet(latent_model_input, t, encoder_hidden_states).sample
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states, attention_mask).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        return self.numpy_to_pil(image)

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image, 'RGB') for image in images]

        return pil_images

    def inception_feature(self, images, device):
        images = torch.stack([self.fid_augment(image) for image in images])
        images = images.type(torch.FloatTensor).to(device)
        images = (images + 1) / 2
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        pred = self.inception(images)[0]

        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        return pred.reshape(-1, 2048)
