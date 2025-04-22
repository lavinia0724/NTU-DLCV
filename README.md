# Deep Learning for Computer Vision by Frank Wang 

## HW1
- Problem 1: Self-supervised pre-training for image classification
- Problem 2: Semantic segmentation
  
Performance:
|                 | p1 val score | p1 test score | p2 val score | p2 test score |
| --------------- |:------------ | ------------- | ------------ |:------------- |
| **My Performance** | **0.5073891626** | **0.4583333333** | **0.75399844** | **0.738113252** |
| Strong Baseline | 0.4           | 0.4           | 0.73         | 0.72          |

## HW2
- Problem 1: Conditional Diffusion Models [digit dataset - MNIST-M & SVHN]
- Problem 2: DDIM [face dataset]
- Problem 3: Personalization [personalization dataset]

Performance:
|                     | p1 acc    | p2 public | p2 private | p3 <new1> prompt0 img2img | p3 <new1> prompt0 txt2img | p3 <new1> prompt1 img2img | p3 <new1> prompt1 txt2img | p3 <new1> private img2img | p3 <new1> private txt2img | p3 <new2> prompt0 img2img | p3 <new2> prompt0 txt2img | p3 <new2> prompt1 img2img | p3 <new2> prompt1 txt2img | p3 <new2> private img2img | p3 <new2> private txt2img |     
| ------------------- | --------- | --------- | ---------- | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | 
| **My Performance** | **0.996** | **2.54** | **2.27** | **82.04** | **35.41** | **75.06** | **35.55** | **84.48** | **30.87** | **70.1** | **32.62** | **70.1** | **31.06** | **72.99** | **31.17** |     |
| Strong Baseline            | 0.95  | 20 (MSE)      | 20  (MSE)      | 79                       | 34                       | 74                       | 35                       | 80                       | 28                       | 70                       | 30                       | 70                       | 30                       | 70                       | 28                       |     

## HW3
- Problem 1: Zero-shot image captioning with LLaVA
- Problem 2: PEFT on Vision and Language Model for Image Captioning
- Problem 3: Visualization of Attention in Image Captioning

Performance:
|                 | p1 public CIDEr | p1 public CLIP | p1 private CIDEr | p1 private CLIP | p2 public CIDEr | p2 public CLIP | p2 private CIDEr | p2 private CLIP |
| --------------- | --------------- | -------------- | ---------------- | --------------- | --------------- | -------------- | ---------------- | --------------- |
| **My performance** | **1.204732275** | **0.775795288** | **1.230375934** | **0.777111206** | **1.095312659** | **0.737560577** | **1.104761291** | **0.742398834** |
| Strong Baseline | 1.14            | 0.77           | 1.17             | 0.77            | 0.94            | 0.73           | 0.96             | 0.73            |

## HW4
- Problem: 3D Novel View Synthesis

Performance:
|                   | public PSNR   | public SSIM    | private PSNR    | private SSIM    |
| :---------------- | :------------ | :------------- | :-------------- | :-------------- |
| **My Performance** | **38.03771405** | **0.9803916465** | **37.88307479** | **0.9799995792** |
|  Baseline   | 35            | 0.97           | 35              | 0.97            |

