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



## &#x1F527; Requirements and Installation

For training and localhosting: Make sure you have access to **GPU A6000** or **A100**

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
## &#x1F527; Dataset Format
```
[
  {
        "id": "1",
        "image": "path/to/image.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nProvide a brief description of the given image."
            },
            {
                "from": "gpt",
                "value": "an apple fall from tree"
            }
        ]
    }
]
```
## &#x1F527; Finetuning

Navaigate to **finetune.sh** and replace the path for your dataset.




## &#x1F527; Transformers

## Launch Gradio Server

```
python tinyllava/serve/app.py --host 0.0.0.0 --port 10000 --model-path "./output/TinyLLaVA-3.1B-lora" --model-name "BU-TinyLLaVA-3.1B-lora"
```

Make sure model-path points to the finetuned model.bin


## &#x270F; Reference

Check out author's work!

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

