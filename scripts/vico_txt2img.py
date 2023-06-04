import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from transformers import CLIPTokenizer

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import PIL
import time

def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    # assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens

def load_imageca(model, image_ca_path):
    state_dict = torch.load(image_ca_path, map_location='cpu')
    state_dict = {"model." + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def image_process(img_path):
    image = Image.open(img_path)

    if not image.mode == "RGB":
        image = image.convert("RGB")
    img = np.array(image).astype(np.uint8)
    image = Image.fromarray(img)
    image = image.resize((512, 512), resample=PIL.Image.BILINEAR)

    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0)
    return image

def run_inference(model, opt, device):
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt

    sample_path = os.path.join(outpath, f'{prompt.replace(" ", "-")}')
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    all_samples=list()
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.n_samples * [""])
            for n in trange(opt.n_iter, desc="Sampling"):
                c = model.get_learned_conditioning(opt.n_samples * [prompt])
                text_tokens = get_clip_token_for_string(tokenizer, opt.n_samples * [prompt])
                ph_tokens = get_clip_token_for_string(tokenizer, opt.n_samples * ["*"])
                ph_tok = ph_tokens[0,1]
                placeholder_idx = torch.where(text_tokens == ph_tok)
                endoftext_idx = (torch.arange(text_tokens.shape[0]), text_tokens.argmax(dim=-1))
                
                if "*" in prompt:
                    # ph_pos = torch.tensor(opt.n_samples * [prompt.strip().split().index("*") + 1]).to(device)
                    ph_pos = [placeholder_idx, endoftext_idx]
                else:
                    # ph_pos = torch.tensor(opt.n_samples * [len(prompt.strip().split()) + 1]).to(device)
                    ph_pos = [endoftext_idx, endoftext_idx]
                
                shape = [4, opt.H//8, opt.W//8]
                
                # get the reference image
                image = image_process(opt.image_path).to(device)
                encoder_posterior = model.encode_first_stage(image)
                xr = model.get_first_stage_encoding(encoder_posterior).detach()
                xr = xr.expand(opt.n_samples, -1, -1, -1)
                
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 image_cond=xr,
                                                 ph_pos=ph_pos,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.jpg"))
                    base_count += 1
                all_samples.append(x_samples_ddim)


    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=opt.n_samples)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.jpg'))

    print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.") 
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txtimg2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="/data/pretrained_models/ldm/text2img-large/model.ckpt", 
        help="Path to pretrained ldm text2img model")

    parser.add_argument(
        "--ft_path", 
        type=str, 
        help="Path to a fine-tuned checkpoint")
    
    parser.add_argument(
        "--load_step",
        type=int,
        default=299,
        help="Training step used to infer"
    )
  
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to a sample image, one image for now."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    # parser.add_argument(
    #     "--embedding_path", 
    #     type=str, 
    #     help="Path to a pre-trained embedding manager checkpoint")

    # parser.add_argument(
    #     "--image_ca_path",
    #     type=str,
    #     help="Path to a pre-trained image cross-attention checkpoint"
    # )
    
    # parser.add_argument(
    #     "--mlp_path",
    #     type=str,
    #     help="Path to a pre-trained image cross-attention checkpoint"
    # )
    
    opt = parser.parse_args()    
    seed_everything(opt.seed)

    image_ca_path = os.path.join(opt.ft_path, "checkpoints/cross_attention-{}.pt".format(opt.load_step))
    embedding_path = os.path.join(opt.ft_path, "checkpoints/embeddings_gs-{}.pt".format(opt.load_step))
    
    print("cross attention path: " + image_ca_path)
    print("embedding path: " + embedding_path)
    
    # config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval_with_tokens.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, opt.ckpt_path)  # TODO: check path
    model.embedding_manager.load(embedding_path)
    
    ############################
    ### add visual embedding
    # state_dict_mlp = torch.load(opt.mlp_path, map_location='cpu')
    # model.embedding_manager.mlp.load_state_dict(state_dict_mlp, strict=True)
    ############################
    
    model = load_imageca(model, image_ca_path)
    model.model.freeze_imageca() # TODO: check it
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt

    sample_path = os.path.join(outpath, f'{prompt.replace(" ", "-")}')
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    all_samples=list()
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.n_samples * [""])
            for n in trange(opt.n_iter, desc="Sampling"):
                c = model.get_learned_conditioning(opt.n_samples * [prompt])
                text_tokens = get_clip_token_for_string(tokenizer, opt.n_samples * [prompt])
                ph_tokens = get_clip_token_for_string(tokenizer, opt.n_samples * ["*"])
                ph_tok = ph_tokens[0,1]
                placeholder_idx = torch.where(text_tokens == ph_tok)
                endoftext_idx = (torch.arange(text_tokens.shape[0]), text_tokens.argmax(dim=-1))
                
                if "*" in prompt:
                    # ph_pos = torch.tensor(opt.n_samples * [prompt.strip().split().index("*") + 1]).to(device)
                    ph_pos = [placeholder_idx, endoftext_idx]
                else:
                    # ph_pos = torch.tensor(opt.n_samples * [len(prompt.strip().split()) + 1]).to(device)
                    ph_pos = [endoftext_idx, endoftext_idx]
                
                shape = [4, opt.H//8, opt.W//8]
                
                # get the reference image
                image = image_process(opt.image_path).to(device)
                encoder_posterior = model.encode_first_stage(image)
                xr = model.get_first_stage_encoding(encoder_posterior).detach()
                xr = xr.expand(opt.n_samples, -1, -1, -1)
                
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 image_cond=xr,
                                                 ph_pos=ph_pos,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.jpg"))
                    base_count += 1
                all_samples.append(x_samples_ddim)


    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=opt.n_samples)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.jpg'))

    print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")