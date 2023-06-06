# ViCo
[![arXiv](https://img.shields.io/badge/arXiv-2306.00971%20-b31b1b)](https://arxiv.org/abs/2306.00971)
![License](https://img.shields.io/github/license/haoosz/ViCo?color=lightgray)

### [**ViCo: Detail-Preserving Visual Condition for Personalized Text-to-Image Generation**](https://arxiv.org/abs/2306.00971)

![teaser](img/teaser.png)

## ⏳ To Do
- [x] Release inference code
- [x] Release pretrained models
- [ ] Release training code
- [ ] Hugging Face demo 

## ⚙️ Set-up
Create a conda environment `vico` using
```
conda env create -f environment.yaml
conda activate vico
```

## ⏬ Download
Download the [pretrained stable diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt) under `models/ldm/stable-diffusion-v1`.

We provide the pretrained checkpoints at 300. 350, and 400 steps of 8 objects. You can download the [**sample images**](https://drive.google.com/drive/folders/1m8TCsY-C1tIOflHtWnFzTbw2C6dq67mC?usp=sharing) and their corresponding [**pretrained checkpoints**](https://drive.google.com/drive/folders/1I9BJpTLEGueK2hCaR2RKdQrlTrtF24lC?usp=drive_link). You can also download the data of any object:

|  Object   | Sample images | Checkpoints |
|  :----  | :----:  | :----:  |
|  barn  | [image](https://drive.google.com/drive/folders/1bS3QYwzAOnOJcdqUNQ4VSGFnlBN87elT?usp=drive_link) | [ckpt](https://drive.google.com/drive/folders/1EsLeRkPUg7WH-nMCept28pVaX0IPlCGu?usp=drive_link) |
|  batman | [image](https://drive.google.com/drive/folders/1S_UFE9mAgaqWHNxrb2XudnuIyWafSwlv?usp=drive_link) | [ckpt](https://drive.google.com/drive/folders/1elwu9CNtzx_hwK23SbJiSfLkpMtbA66d?usp=drive_link) |
|  clock  | [image](https://drive.google.com/drive/folders/1L4AqVO0o6dapAxjjfSUCVGwd9iB5hIv2?usp=drive_link)  |  [ckpt](https://drive.google.com/drive/folders/1N0E-he1GLH_3c-H1E8204xYzOKU-RT_X?usp=drive_link)  |
|  dog7  | [image](https://drive.google.com/drive/folders/107YOi1qXHnGeDuAaxxe4AW9fj17hehxX?usp=drive_link)   |  [ckpt](https://drive.google.com/drive/folders/1SujoFfOBeKbZI74mFrdCsDIov_5xprHb?usp=drive_link)  |
|  monster toy  |  [image](https://drive.google.com/drive/folders/18nIAXQsG5KaGys2yNJtIuYso2cgZh-2f?usp=drive_link)  |  [ckpt](https://drive.google.com/drive/folders/1EzDjyyya7_zOflOG5rPkxY--R5OxejYx?usp=drive_link)   |
|  pink sunglasses  |  [image](https://drive.google.com/drive/folders/10it3Sd9U1wbkfksMWfFHXeAch6uanEDr?usp=drive_link)  |   [ckpt](https://drive.google.com/drive/folders/1aHnAgM4dpWFsqiNeg3mIX68G6xjfuZ-X?usp=drive_link)   |
|  teddybear  |  [image](https://drive.google.com/drive/folders/1lT8mOSgeh0P8DlfIh34qC2cvk2QaqSBo?usp=drive_link)  |  [ckpt](https://drive.google.com/drive/folders/1630qFd06T2Kz46pb-hs9OA99v3LD44IQ?usp=drive_link)   |
|  wooden pot  |  [image](https://drive.google.com/drive/folders/1eVDMNAfAEroqMV8AiFlBqRGNcElmWw70?usp=drive_link)  |  [ckpt](https://drive.google.com/drive/folders/1kXQuzfSsAJ895gHZJDiFF-5BHoX49gOx?usp=drive_link)    |

Datasets are originally collected and provided by [Textual Inversion](https://github.com/rinongal/textual_inversion), [DreamBooth](https://github.com/google/dreambooth), and [Custom Diffsuion](https://github.com/adobe-research/custom-diffusion).

## 🚀 Inference
Before run the inference command, please set:  
- `REF_IMAGE_PATH`: Path of **the reference image**. It can be any image in the samples like `batman/1.jpg`.
- `CHECKPOINT_PATH`: Path of **the checkpoint weight**. Its 
subfolder should be similar to `checkpoints/*-399.pt`.
- `OUTPUT_PATH`: Path of **the generated images**. For example, it can be like `outputs/batman`.
```
python scripts/vico_txt2img.py \
--ddim_eta 0.0  --n_samples 4  --n_iter 2  --scale 7.5  --ddim_steps 50  \
--ckpt_path models/ldm/stable-diffusion-v1/sd-v1-4.ckpt  \
--image_path REF_IMAGE_PATH \
--ft_path CHECKPOINT_PATH \
--load_step 399 \
--prompt "a photo of * on the beach" \
--outdir OUTPUT_PATH
```
You can specify `load_step` (300,350,400) and personalize `prompt` (a prefix "a photo of" usually makes better results).

## 💻 Training
### Coming soon!

## 📖 Citation
If you use this code in your research, please consider citing our paper:
```bibtex
@inproceedings{Hao2023ViCo,
  title={ViCo: Detail-Preserving Visual Condition for Personalized Text-to-Image Generation},
  author={Shaozhe Hao and Kai Han and Shihao Zhao and Kwan-Yee K. Wong},
  year={2023}
}
```

## Acknowledgements
This code repository is based on the great work of [Textual Inversion](https://github.com/rinongal/textual_inversion). Thanks!