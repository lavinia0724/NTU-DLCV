# Deep Learning for Computer Vision by Frank Wang 

## HW1
- Problem 1: Self-supervised pre-training for image classification
- Problem 2: Semantic segmentation
  
Performance:
|                 | p1_val_score | p1_test_score | p2_val_score | p2_test_score |
| --------------- |:------------ | ------------- | ------------ |:------------- |
| **My Performance** | **0.5073891626** | **0.4583333333** | **0.75399844** | **0.738113252** |
| Strong Baseline | 0.4           | 0.4           | 0.73         | 0.72          |

## HW2
- Problem 1: Conditional Diffusion Models [digit dataset - MNIST-M & SVHN]
- Problem 2: DDIM [face dataset]
- Problem 3: Personalization [personalization dataset]

Performance:
|                     | p1_acc    | p2_public | p2_private | p3_<new1>prompt0_img2img | p3_<new1>prompt0_txt2img | p3_<new1>prompt1_img2img | p3_<new1>prompt1_txt2img | p3_<new1>private_img2img | p3_<new1>private_txt2img | p3_<new2>prompt0_img2img | p3_<new2>prompt0_txt2img | p3_<new2>prompt1_img2img | p3_<new2>prompt1_txt2img | p3_<new2>private_img2img | p3_<new2>private_txt2img |     |
| ------------------- | --------- | --------- | ---------- | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | --- |
| **My Performance** | **0.996** | **2.54** | **2.27** | **82.04** | **35.41** | **75.06** | **35.55** | **84.48** | **30.87** | **70.1** | **32.62** | **70.1** | **31.06** | **72.99** | **31.17** |     |
| Baseline            | 0.9/0.95  | 20 (MSE)      | 20  (MSE)      | 79                       | 34                       | 74                       | 35                       | 80                       | 28                       | 70                       | 30                       | 70                       | 30                       | 70                       | 28                       |     |

## HW3

