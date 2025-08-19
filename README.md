# Make Me Happier: Evoking Emotions Through Image Diffusion Models

## [Paper](https://arxiv.org/pdf/2403.08255.pdf) | [Dataset](https://drive.google.com/drive/folders/1aaZUNo-domsSOH_2_flRU8LMZjptPU28?usp=drive_link) | [Models](https://drive.google.com/drive/folders/1Wkrq5d3JC96PecGJQliz7h5cUmfACpqi?usp=sharing)
PyTorch implementation of EmoEditor, an emotion-evoked diffusion model.  
This work has been accepted to ICCV 2025.

**[Make Me Happier: Evoking Emotions Through Image Diffusion Models](https://arxiv.org/pdf/2403.08255.pdf)**  
Qing Lin, [Jingfeng Zhang](https://zjfheart.github.io/), [Yew-Soon Ong](https://www3.ntu.edu.sg/home/asysong/), [Mengmi Zhang](https://a0091624.wixsite.com/deepneurocognition-1)*  
*Corresponding author  

## Abstract
Despite the rapid progress in image generation, emotional image editing remains under-explored. The semantics, context, and structure of an image can evoke emotional responses, making emotional image editing techniques valuable for various real-world applications, including treatment of psychological disorders, commercialization of products, and artistic design. First, we present a novel challenge of emotion-evoked image generation, aiming to synthesize images that evoke target emotions while retaining the semantics and structures of the original scenes. To address this challenge, we propose a diffusion model capable of effectively understanding and editing source images to convey desired emotions and sentiments. Moreover, due to the lack of emotion editing datasets, we provide a unique dataset consisting of 340,000 pairs of images and their emotion annotations. Furthermore, we conduct human psychophysics experiments and introduce a new evaluation metric to systematically benchmark all the methods. Experimental results demonstrate that our method surpasses all competitive baselines. Our diffusion model is capable of identifying emotional cues from original images, editing images that elicit desired emotions, and meanwhile, preserving the semantic structure of the original images.

<div align=left><img src="./fig/fig1_teaser.png" width="99%" height="99%" ></div>  
<!-- The generated images evoke a sense of happiness in viewers, contrasting with the negative emotions elicited by the source images. Given a source image that triggers negative emotions (framed in green), our method (Ours) synthesizes a new image that elicits the given positive target emotions (in red), while maintaining the essential elements and structures of the scene. For instance, our method replaces the anger-inducing flames in the source image with cute-shaped lamps to evoke the target emotion of amusement. While in an outdoor setting, the raging fire is substituted with a tranquil, lush meadow to inspire a sense of awe. For comparisons, we include other competitive methods. The blue number below each image represents its CAM-based ESMI score, with higher values being better.  -->

## Generalization to Real-world Scenarios
<div align=left><img src="./fig/fig12_pos2pos.png" width="99%" height="99%" ></div>

## EmoPair Dataset
<div align=left><img src="./fig/fig4_dataset.png" width="99%" height="99%" ></div>  
The dataset comprises two subsets: EmoPair-Annotated Subset (EPAS, left blue box) and EmoPair-Generated Subset (EPGS, right orange box). Each subset includes schematics depicting the creation, selection, and labeling of image pairs in the upper quadrants, with two example pairs in the lower quadrants. Each example pair comprises a source image (framed in green) and a target image. The classified source and target emotion labels (highlighted in red) and target-emotion-driven text instructions for image editing are provided.

* Download our [EPGS dataset](https://drive.google.com/drive/folders/10jwTjzVpTLTOe8HnzgLBgtyFvYPMlybb?usp=drive_link) and put it in `EmoPair/EPGS/`.
* Download [Ip2p images](https://instruct-pix2pix.eecs.berkeley.edu/clip-filtered-dataset/) and put them in `EmoPair/EPAS/ip2p_clip/`.
* The dataset annotation can be found in [data_EmoPair.json](https://drive.google.com/file/d/1aEHnfqVPwtey6zv3cRnmQvbtWhF5c5Z-/view?usp=drive_link).

## Environment Setup
```
conda create -n emoeditor python=3.10
conda activate emoeditor 

pip install torch torchvision torchaudio
pip install diffusers
pip install accelerate
pip install transformers==4.49
pip install opencv-python
pip install scikit-image
pip install grad-cam
```

## Training & Testing



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
