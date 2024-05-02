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

