# Multimodal-Context-Reasoning
<p> 
  <a href="https://scholar.google.com/citations?user=U98QY0QAAAAJ&hl=en"><img src="https://img.shields.io/badge/scholar-4385FE.svg?&style=plastic&logo=google-scholar&logoColor=white" alt="Google Scholar" height="18px"> </a>
  <a href="https://twitter.com/LyxTg"> <img src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" height="18px" alt="Yunxin Li">
</p> 
A multimodal context reasoning approach that introduce the multi-view semantic alignment information via prefix tuning.

Our paper "A Multi-Modal Context Reasoning Approach for Conditional Inference on Joint Textual and Visual Clues" has been accepted by ACL 2023 Main Conference.

## Requirement
Python >= 3.8.10, 
torch >= 1.10.0

For some .zip files, you should unzip them in the current path, which contain some modified codes.

For the used Oscar-base version and pretrained phrase-level alignment model, you can download them from the [HuggingFace Hub](https://huggingface.co/YunxinLi)

## Training

For PMR task, you can run the file:
```
python ModCR_PMR.py
```
  
For VCR task, you can run the file:
```
python vcr_modcr.py
```

## Acknowledge

Thanks all contributors for their supports!!!

If you're using LMEye in your research or applications, please cite our work:

```
@article{li2023multi,
  title={A Multi-Modal Context Reasoning Approach for Conditional Inference on Joint Textual and Visual Clues},
  author={Li, Yunxin and Hu, Baotian and Chen, Xinyu and Ding, Yuxin and Ma, Lin and Zhang, Min},
  journal={arXiv preprint arXiv:2305.04530},
  year={2023}
}
```
