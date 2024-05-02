<h2 align="center"> <a href="https://arxiv.org/abs/2402.14289">TinyLLaVA: A Framework of Small-scale Large Multimodal Models</a>

<h5 align="center">


![image](https://github.com/weibb123/BU_TinyLLava/assets/84426364/587b7a11-1e20-4379-a122-b2f7cf249fe6)

# DEMO on Gradio Server
https://www.youtube.com/watch?v=Hl90tcNGGjY

## Brief Description

Engineered an AI-powered campus tour guide for Boston University by fine-tuning TinyLlava (small multimodal model) to deliver tailored, interactive experiences.

Leveraged Ollama to design a Streamlit application integrating Llava Vision with dynamic chat functionality, improving user engagement.

make use of vision transformer to classify images to recognize BU image buildings.



## Contents

- [Install](#x1f527-requirements-and-installation)
- [Model Zoo](#x1f433-model-zoo)
- [Demo](#Demo)
- [Quick Start](#x1f527-quick-start)
- [Run Inference](#x1f527-run-inference)
- [Evaluation](#evaluation)
- [Data](#data-preparation)
- [Train](#train)
- [Custom Finetune](#custom-finetune)


## &#x1F527; Requirements and Installation

We recommend the requirements as follows.

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/DLCV-BUAA/TinyLLaVABench.git
cd TinyLLaVABench
```

2. Install Package
```Shell
conda create -n tinyllava python=3.10 -y
conda activate tinyllava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```Shell
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
### Upgrade to the latest code base

```Shell
git pull
pip install -e .

# if you see some import errors when you upgrade, please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir
```

## &#x1F433; Model Zoo
### Legacy Model
- [tiny-llava-hf](https://huggingface.co/bczhou/tiny-llava-v1-hf)

### Pretrained Models
- [TinyLLaVA-3.1B](https://huggingface.co/bczhou/TinyLLaVA-3.1B)
- [TinyLLaVA-2.0B](https://huggingface.co/bczhou/TinyLLaVA-2.0B)
- [TinyLLaVA-1.5B](https://huggingface.co/bczhou/TinyLLaVA-1.5B)

### Model Details
| Name          | LLM               | Checkpoint                                     | LLaVA-Bench-Wild | MME      | MMBench | MM-Vet | SQA-image | VQA-v2 | GQA   | TextVQA |
|---------------|-------------------|------------------------------------------------|------------------|----------|---------|--------|-----------|--------|-------|---------|
| TinyLLaVA-3.1B | Phi-2             | [TinyLLaVA-3.1B](https://huggingface.co/bczhou/TinyLLaVA-3.1B) | 75.8             | 1464.9   | 66.9    | 32.0   | 69.1      | 79.9   | 62.0  | 59.1    |
| TinyLLaVA-2.0B | StableLM-2-1.6B   | [TinyLLaVA-2.0B](https://huggingface.co/bczhou/TinyLLaVA-2.0B) | 66.4             | 1433.8     | 63.3    | 32.6   | 64.7      | 78.9   | 61.9  | 56.4    |
| TinyLLaVA-1.5B | TinyLlama         | [TinyLLaVA-1.5B](https://huggingface.co/bczhou/TinyLLaVA-1.5B) | 60.8             | 1276.5     | 55.2     | 25.8   | 60.3      | 76.9   | 60.3  | 51.7    |


## Demo

### Gradio Web Demo

Launch a local web demo by running:
```shell
python tinyllava/serve/app.py --model-path bczhou/TinyLLaVA-3.1B --model-name TinyLLaVA-3.1B
```

### CLI Inference

We also support running inference with CLI. To use our model, run:
```shell
python -m tinyllava.serve.cli \
    --model-path bczhou/TinyLLaVA-3.1B \
    --image-file "./tinyllava/serve/examples/extreme_ironing.jpg" 
```


## &#x1F527; Quick Start

<details>
<summary>Load model</summary>
    
```Python
from tinyllava.model.builder import load_pretrained_model
from tinyllava.mm_utils import get_model_name_from_path
from tinyllava.eval.run_tiny_llava import eval_model

model_path = "bczhou/TinyLLaVA-3.1B"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
```
</details>

## &#x1F527; Run Inference
Here's an example of running inference with [TinyLLaVA-3.1B](https://huggingface.co/bczhou/TinyLLaVA-3.1B)
<details>
<summary>Run Inference</summary>
    
```Python
from tinyllava.model.builder import load_pretrained_model
from tinyllava.mm_utils import get_model_name_from_path
from tinyllava.eval.run_tiny_llava import eval_model

model_path = "bczhou/TinyLLaVA-3.1B"
prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": "phi",
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)
```
</details>

### Important
We use different `conv_mode` for different models. Replace the `conv_mode` in `args` according to this table:
| model          	| conv_mode 	|
|----------------	|-----------	|
| TinyLLaVA-3.1B 	| phi       	|
| TinyLLaVA-2.0B 	| phi       	|
| TinyLLaVA-1.5B 	| v1        	|

## Evaluation
To ensure the reproducibility, we evaluate the models with greedy decoding.

See [Evaluation.md](https://github.com/DLCV-BUAA/TinyLLaVABench/blob/main/docs/Evaluation.md)

## Data Preparation

In our paper, we used two different datasets: the [LLaVA dataset](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#pretrain-feature-alignment) and the [ShareGPT4V dataset](https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md), and compared their differences. In this section, we provide information on data preparation.

### Pretraining Images
* LLaVA: The pretraining images of LLaVA is from the 558K subset of the LAION-CC-SBU dataset.
* ShareGPT4V: The pretraining images of ShareGPT4V is a mixture of 558K LAION-CC-SBU subset, SAM dataset, and COCO dataset.

### Pretraining Annotations
* LLaVA: The pretraining annotations of LLaVA are [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).
* ShareGPT4V: The pretraining annotations of ShareGPT4V are [here](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/share-captioner_coco_lcs_sam_1246k_1107.json).


### SFT Images & Annotations
The majority of the two SFT datasets are the same, with the exception that the 23K detailed description data in LLaVA-1.5-SFT being replaced with detailed captions randomly sampled from the [100K ShareGPT4V data](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/sharegpt4v_instruct_gpt4-vision_cap100k.json).

### Download data

1. Download relevant images

- LAION-CC-SBU-558K: [images.zip](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/images.zip)
- COCO: This dataset is from the [COCO2017 challenge](https://cocodataset.org/). Download: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- WebData: This dataset is curated by the [ShareGPT4V project](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V). Download: [images](https://drive.google.com/drive/folders/1tCUQ-sq6vdshZVkF0ZeF3K4eztkXJgax?usp=sharing). Only for academic usage.
- SAM: This dataset is collected by [Meta](https://ai.meta.com/datasets/segment-anything-downloads/). Download: [images](https://ai.meta.com/datasets/segment-anything-downloads/). We only use 000000~000050.tar for now. If you just want to use ShareGPT4V for SFT, you can quickly download 9K images from [here](https://drive.google.com/file/d/1dKumdOKSXtV7lIXdrG7jsIK_z2vZv2gs/view?usp=drive_link).
- GQA: [GQA project page](https://cs.stanford.edu/people/dorarad/gqa/about.html). Download: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [OCR-VQA project page](https://ocr-vqa.github.io/). Download: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing). We save all files as `.jpg`
- TextVQA: [TextVQA project page](https://textvqa.org/). Download: [trainvalimages](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [VisualGenome project page](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html). Download: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)


2. Download relevant annotations

- LLaVA's pretraining annotations: [blip_laion_cc_sbu_558k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)
- LLaVA's SFT annotations: [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)
- ShareGPT4V's pretraining annotations: [share-captioner_coco_lcs_sam_1246k_1107.json](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/share-captioner_coco_lcs_sam_1246k_1107.json)
- ShareGPT4V's SFT annotations: [sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json)


### Organize Data

Organize the image files and annotation files as follows in `path/to/your/data`:

```none
data
├── llava
│   ├── llava_pretrain
│   │   ├── images
│   │   ├── blip_laion_cc_sbu_558k.json
├── coco
│   ├── train2017
├── sam
│   ├── images
├── gqa
│   ├── images
├── ocr_vqa
│   ├── images
├── textvqa
│   ├── train_images
├── vg
│   ├── VG_100K
│   ├── VG_100K_2
├── share_textvqa
│   ├── images
├── web-celebrity
│   ├── images
├── web-landmark
│   ├── images
├── wikiart
│   ├── images
├── text_files
│   ├── llava_v1_5_mix665k.json
│   ├── share-captioner_coco_lcs_sam_1246k_1107.json
│   ├── sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json
```

## Train

**This section we describe the base recipe.**
### Hyperparameters
Both hyperparameters used in pretraining and finetuning are provided below.

1. Pretraining

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
|----------------| ---: | ---: | ---: |-----------:| ---: |
| TinyLLaVA-3.1B | 256 | 1e-3 | 1 |       3072 | 0 |

2. Finetuning

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
|----------------| ---: | ---: | ---: |-----------:| ---: |
| TinyLLaVA-3.1B | 128 | 2e-5 | 1 |       3072 | 0 |

### Pretrain

**Replace paths to your paths**

Training script with DeepSpeed ZeRO-2: [`pretrain.sh`](https://github.com/DLCV-BUAA/TinyLLaVABench/blob/main/scripts/tiny_llava/pretrain.sh).

### Finetune

**Replace paths to your paths**

Training script with DeepSpeed ZeRO-3: [`finetune.sh`](https://github.com/DLCV-BUAA/TinyLLaVABench/blob/main/scripts/tiny_llava/finetune.sh).

## Custom-Finetune

Check out our custom finetune using LoRA [here](https://github.com/DLCV-BUAA/TinyLLaVABench/blob/dev/docs/CUTOM_FINETUNE.md).


## &#x270F; Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```BibTeX
@misc{zhou2024tinyllava,
      title={TinyLLaVA: A Framework of Small-scale Large Multimodal Models}, 
      author={Baichuan Zhou and Ying Hu and Xi Weng and Junlong Jia and Jie Luo and Xien Liu and Ji Wu and Lei Huang},
      year={2024},
      eprint={2402.14289},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


## ❤️ Community efforts
* Our codebase is built upon the [LLaVA](https://github.com/haotian-liu/LLaVA) project. Great work!
* Our project uses data from the [ShareGPT4V](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V) project. Great work!

