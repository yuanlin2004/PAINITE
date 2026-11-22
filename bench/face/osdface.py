"""OSDFace (Wang et al., CVPR 2025): one-step diffusion face restoration on SD 2.1-base.

Mirrors upstream infer.py (LoRA merged into the UNet, LQ image embedding as the prompt,
one UNet step at t=399) without its multiprocessing/CLI. The VAE latent uses the posterior
mean instead of a sample so output is deterministic.
"""
import sys
from types import SimpleNamespace

import torch

from restorers import FACE_WEIGHTS, SRC_DIR, Restorer

T = 399
LORA_ALPHA_OVER_RANK = 1.0  # --lora_alpha 16 --lora_rank 16 upstream


def _merge_lora(unet, path):
    from safetensors.torch import load_file

    lora = load_file(path)
    sd = unet.state_dict()
    for ka in [k for k in lora if "lora_A" in k]:
        a, b = lora[ka], lora[ka.replace("lora_A", "lora_B")]
        key = ka.replace(".lora_A.weight", ".weight").replace("unet.", "")
        delta = (b.flatten(1) @ a.flatten(1)).view(sd[key].shape)
        sd[key] = sd[key] + LORA_ALPHA_OVER_RANK * delta.to(sd[key].dtype)
    unet.load_state_dict(sd)


class OSDFace(Restorer):
    def __init__(self, dtype=torch.float16):
        sys.path.insert(0, str(SRC_DIR / "OSDFace"))
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        from lq_embed import TwoLayerConv1x1, vqvae_encoder

        sd = FACE_WEIGHTS / "sd21base"
        w = FACE_WEIGHTS / "osdface"
        self.dtype = dtype
        self.alphas = DDIMScheduler.from_pretrained(sd, subfolder="scheduler").alphas_cumprod.cuda().double()
        self.vae = AutoencoderKL.from_pretrained(sd, subfolder="vae", variant="fp16").cuda().to(dtype).eval()
        unet = UNet2DConditionModel.from_pretrained(sd, subfolder="unet", variant="fp16", torch_dtype=torch.float32)
        _merge_lora(unet, w / "pytorch_lora_weights.safetensors")
        self.unet = unet.cuda().to(dtype).eval()
        args = SimpleNamespace(cat_prompt_embedding=False, use_pos_embedding=False, use_att_pool=False,
                               learnable_pos_emb=False, img_encoder_weight=str(w / "associate_2.ckpt"))
        self.enc = vqvae_encoder(args).cuda().to(dtype).eval()
        self.emb = TwoLayerConv1x1(512, 1024)
        self.emb.load_state_dict(torch.load(w / "embedding_change_weights.pth", weights_only=False))
        self.emb.cuda().to(dtype).eval()

    def __call__(self, x):
        # the LQ encoder's quantizer only handles one image at a time
        return torch.cat([self._one(x[i:i + 1]) for i in range(x.shape[0])])

    @torch.inference_mode()
    def _one(self, x):
        lq = (x * 2 - 1).to(self.dtype)
        prompt = self.emb(self.enc(lq).reshape(x.shape[0], 77, -1))
        sf = self.vae.config.scaling_factor
        z = self.vae.encode(lq).latent_dist.mean * sf
        eps = self.unet(z, T, encoder_hidden_states=prompt).sample
        a = self.alphas[T]
        x0 = ((z.double() - (1 - a).sqrt() * eps.double()) / a.sqrt()).to(self.dtype)
        y = self.vae.decode(x0 / sf).sample.float().clamp(-1, 1)
        return (y * 0.5 + 0.5).clamp(0, 1)
