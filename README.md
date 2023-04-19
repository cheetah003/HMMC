<div style="text-align:center"><h1>HMMC<br/></h1>
<h2>End-to-end Pre-training with Hierarchical Matching and Momentum Contrast for Text-Video Retrieval</h2>
</div>

<p style="text-align:center">
<a href="https://huggingface.co/spaces/cheetah003/HMMC_t2v_search" target="_blank">Try Demo here</a>
</p>

<p style="text-align:center">
  <a href="https://www.python.org/" target="_blank">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python"/>
  </a>
  <a href="https://pytorch.org/" target="_blank">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch"/>
  </a>
  <a href="https://github.com/cheetah003/HMMC/stargazers">
    <img src="https://img.shields.io/github/stars/cheetah003/HMMC?logo=github&style=for-the-badge" alt="Github stars"/>
  </a>
  <a href="https://github.com/cheetah003/HMMC/network/members">
    <img src="https://img.shields.io/github/forks/cheetah003/HMMC?logo=github&style=for-the-badge" alt="Github forks"/>
  </a>
  <a href="https://huggingface.co/spaces/cheetah003/HMMC_t2v_search" target="_blank">
    <img src="https://img.shields.io/badge/dynamic/json?style=for-the-badge&label=Hugging%20Face%20Space&query=%24.runtime.stage&url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fspaces%2Fcheetah003%2FHMMC_t2v_search" alt="Demo"/>
  </a>
</p>

The implementation of paper "End-to-end Pre-training with Hierarchical Matching and Momentum Contrast for Text-Video Retrieval".

HMMC(Hierarchical Matching and Momentum Contrast) is a text-video retrieval model(support Chinese and English) based on [CLIP](https://github.com/openai/CLIP), which pre-trained on 400M image-text pairs in an end-to-end manner. We introduce HMMC model for video-language pre-training, taking advantage of both global video representation and frame features with a hierarchical matching mechanism. We also collected a large-scale Chinese video-language dataset (over 763k unique videos) named CHVTT to explore the multilevel semantic connections between videos and texts. Experimental results on two major Text-video retrieval benchmark datasets demonstrate the advantages of our methods.


## Model Architecture
### Overall Architecture: ###
![Architecture](pics/model.png)
#### Hierarchical Matching: ####
![HM](pics/HM_train.png)


## Requirement
```
pip install -r requirements.txt
```

## Data Preparing

### Public Datasets ###
* MSR-VTT [download link](http://ms-multimedia-challenge.com/2017/dataset)

* VATEX(Chinese and English version) [download link](https://eric-xw.github.io/vatex-website/download.html)

### Write videos to lmdb ###



## Visualization
### Results: ###
![results](pics/visualHM.png)
### Attention map: ###
![Attention](pics/visual_attention1.png)