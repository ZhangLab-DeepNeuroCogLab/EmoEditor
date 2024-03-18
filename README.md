# Make Me Happier: Evoking Emotions Through Image Diffusion Models

## [Demo]() | [Paper](https://arxiv.org/pdf/2403.08255.pdf) | [Data]()
PyTorch implementation of EmoEditor, an emotion-evoked diffusion model.  
Code and data will be released upon paper acceptance.

**[Make Me Happier: Evoking Emotions Through Image Diffusion Models](https://arxiv.org/pdf/2403.08255.pdf)**  
Qing Lin, [Jingfeng Zhang](https://zjfheart.github.io/), [Yew Soon Ong](https://www3.ntu.edu.sg/home/asysong/), [Mengmi Zhang](https://a0091624.wixsite.com/deepneurocognition-1)*  
*Corresponding author  
<div align=left><img src="./fig/fig1_teaser.png" width="90%" height="90%" ></div>  
The generated images evoke a sense of happiness in viewers, contrasting with the negative emotions elicited by the source images. Given a source image that triggers negative emotions (framed in green), our method (Ours) synthesizes a new image that elicits the given positive target emotions (in red), while maintaining the essential elements and structures of the scene.

## Results in Cross-Valence Scenarios 
<div align=left><img src="./fig/fig9_visualization.png" width="90%" height="90%" ></div>  

## Results in Same-Valence Scenarios 
<div align=left><img src="./fig/fig12_pos2pos.png" width="90%" height="90%" ></div>

## Results in Neutral-Valence Scenarios 
<div align=left><img src="./fig/fig11_neutral.png" width="90%" height="90%" ></div>

## EmoPair Dataset
<div align=left><img src="./fig/fig4_dataset.png" width="90%" height="90%" ></div>  

## Benchmark
human psychophysics experiments and four newly introduced metrics
<div align=left><img src="./fig/fig6_user_study.png" width="50%" height="50%" ></div>  

| **Method** | **EMR(%)↑** | **ESR(%)↑** | **ENRD↓** | **ESS↓** |
|:---|:---:|:---:|:---:|:---:|
| **CT** | 6.89 | 79.32 | 32.61 | **7.36** |
| **NST** | 34.42 | 92.01 | 34.42 | 18.57 |
| **Csty** | 11.51 | 85.52 | 41.59 | 36.64 |
| **Ip2p** | 2.53 | 67.76 | **9.82** | <ins>12.71</ins> |
| **LMS** | 11.51 | 77.38 | 26.67 | 19.74 |
| **w/o $\mathcal{P}$** | 5.06 | 69.15 | <ins>20.08</ins> | 14.93 |
| **w/o $L_{emb}$** | <ins>41.12</ins> | <ins>92.36</ins> | 23.02 | 16.06 |
| **Ours** | **50.20** | **92.86** | 24.73 | 16.27 |



## BibTeX
```
@article{lin2024emoeditor,
  title={Make Me Happier: Evoking Emotions Through Image Diffusion Models},
  author={Qing Lin and Jingfeng Zhang and Yew Soon Ong and Mengmi Zhang},
  journal={arXiv preprint arXiv:2403.08255},
  year={2024}
}
```


## Acknowledgments
We benefit a lot from [CompVis/stable_diffusion](https://github.com/CompVis/stable-diffusion), [timothybrooks/instruct-pix2pix](https://github.com/timothybrooks/instruct-pix2pix/tree/main?tab=readme-ov-file) and [ayaanzhaque/instruct-nerf2nerf](https://github.com/ayaanzhaque/instruct-nerf2nerf) repo.
