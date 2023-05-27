# Awesome-Diffusion-Learning-Repository
This repository contains a collection of papers on ***Diffusion Learning***.

基于扩散原理的生成模型旨在将先验数据分布转化为随机噪声，然后再一步一步地修正转换重建一个全新的样本，其不需要像VAE那样对齐后验分布，也不需要训练额外的判别器（如GAN），或施加网络约束（如归一化流）。由于上述优点，扩散学习已经在计算机视觉、自然语言处理和图分析等领域引起广泛关注。扩散模型提供了概率参数化的可操作的描述模型，有足够理论支持的稳定训练程序和统一的损失函数设计，具有较高的实用价值。如果想要了解更多关于生成式扩散模型的算法和应用，可以查阅[[Link](https://github.com/chq1155/A-Survey-on-Generative-Diffusion-Model)] 

![idea](https://github.com/chenshuoXu2002/Awesome-Diffusion-Learning-Repository/assets/134117808/98df610f-7581-4dc8-8d36-29e99edbdab8)


## Contents
- [Resources](#resources)
  - [Introductory Papers](#introductory-papers)
- [Papers](#papers)
  - [Survey](#survey)
  - [Vision](#vision)
    - [Generation](#generation)
    - [Classification](#classification)
    - [Segmentation](#segmentation)
    - [Image Translation](#image-translation)
    - [Inverse Problems](#inverse-problems)
    - [Medical Imaging](#medical-imaging)
    - [Multi-modal Learning](#multi-modal-learning)
    - [3D Vision](#3d-vision)
    - [Adversarial Attack](#adversarial-attack)
  - [Audio](#audio)
    - [Generation](#generation-1)
    - [Conversion](#conversion)
    - [Enhancement](#enhancement)
    - [Separation](#separation)
    - [Text-to-Speech](#text-to-speech)
  - [Natural Language](#natural-language)
  - [Tabular and Time Series](#tabular-and-time-series)
    - [Generation](#generation-2)
    - [Forecasting](#forecasting)
    - [Imputation](#imputation)
  - [Graph](#graph)
    - [Generation](#generation-3)
    - [Molecular and Material Generation](#molecular-and-material-generation)
  - [Theory](#theory)
  - [Applications](#applications)


# Resources

## Introductory Papers


**Understanding Diffusion Models: A Unified Perspective** \
*Calvin Luo* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.11970)] \
25 Aug 2022

**How to Train Your Energy-Based Models** \
*Yang Song, Diederik P. Kingma* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2101.03288)] \
9 Jan 2021



# Papers



## Survey


**Diffusion Models for Medical Image Analysis: A Comprehensive Survey** \
*Amirhossein Kazerouni, Ehsan Khodapanah Aghdam, Moein Heidari, Reza Azad, Mohsen Fayyaz, Ilker Hacihaliloglu, Dorit Merhof* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.07804)] [[Github](https://github.com/amirhossein-kz/Awesome-Diffusion-Models-in-Medical-Imaging)] \
14 Nov 2022


## Vision


### Generation


**LEO: Generative Latent Image Animator for Human Video Synthesis** \
*Yaohui Wang, Xin Ma, Xinyuan Chen, Antitza Dantcheva, Bo Dai, Yu Qiao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03989)] [[Project](https://wyhsirius.github.io/LEO-project/)] [[Github](https://github.com/wyhsirius/LEO)] \
6 May 2023

**DreamPose: Fashion Image-to-Video Synthesis via Stable Diffusion** \
*Johanna Karras, Aleksander Holynski, Ting-Chun Wang, Ira Kemelmacher-Shlizerman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06025)] [[Project](https://grail.cs.washington.edu/projects/dreampose/)] [[Github](https://github.com/johannakarras/DreamPose)] \
12 Apr 2023

**UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models** \
*Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, Jiwen Lu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.04867)] [[Project](https://unipc.ivg-research.xyz)] [[Github](https://github.com/wl-zhao/UniPC)] \
9 Feb 2023

**Scalable Diffusion Models with Transformers** \
*William Peebles, Saining Xie* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09748)] [[Project](https://www.wpeebles.com/DiT)] [[Github](https://github.com/facebookresearch/DiT)] \
19 Dec 2022

**VIDM: Video Implicit Diffusion Models** \
*Kangfu Mei, Vishal M. Patel* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.00235)] [[Project](https://kfmei.page/vidm/)] [[Github](https://github.com/MKFMIKU/VIDM)] \
1 Dec 2022

**Seeing Beyond the Brain: Conditional Diffusion Model with Sparse Masked Modeling for Vision Decoding** \
*Zijiao Chen, Jiaxin Qing, Tiange Xiang, Wan Lin Yue, Juan Helen Zhou* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.06956)] [[Project](https://mind-vis.github.io/)] [[Github](https://github.com/zjc062/mind-vis)] \
13 Nov 2022

**GENIE: Higher-Order Denoising Diffusion Solvers** \
*Tim Dockhorn, Arash Vahdat, Karsten Kreis* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2210.05475)] [[Project](https://nv-tlabs.github.io/GENIE/) [[Github](https://github.com/nv-tlabs/GENIE)] \
11 Oct 2022

**Diffusion Autoencoders: Toward a Meaningful and Decodable Representation** \
*Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, Supasorn Suwajanakorn* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2111.15640)] [[Project](https://diff-ae.github.io/)] [[Github](https://github.com/phizaz/diffae)] \
30 Dec 2021

**SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations** \
*Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, Stefano Ermon* \
ICLR  2022. [[Paper](https://arxiv.org/abs/2108.01073)] [[Project](https://sde-image-editing.github.io/)] [[Github](https://github.com/ermongroup/SDEdit)] \
2 Aug 2021

**D2C: Diffusion-Denoising Models for Few-shot Conditional Generation** \
*Abhishek Sinha<sup>1</sup>, Jiaming Song<sup>1</sup>, Chenlin Meng, Stefano Ermon* \
NeurIPS 2021. [[Paper](https://arxiv.org/abs/2106.06819)] [[Project](https://d2c-model.github.io/)] [[Github](https://github.com/d2c-model/d2c-model.github.io)] \
12 Jun 2021

**Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling** \
*Valentin De Bortoli, James Thornton, Jeremy Heng, Arnaud Doucet* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.01357)] [[Project](https://jtt94.github.io/papers/schrodinger_bridge)] [[Github](https://github.com/JTT94/diffusion_schrodinger_bridge)] \
1 Jun 2021

**Image Super-Resolution via Iterative Refinement** \
*Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, Mohammad Norouzi* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2104.07636)] [[Project](https://iterative-refinement.github.io/)] [[Github](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)] \
15 Apr 2021 

**Generative Modeling by Estimating Gradients of the Data Distribution** \
*Yang Song, Stefano Ermon* \
NeurIPS 2019. [[Paper](https://arxiv.org/abs/1907.05600)] [[Project](https://yang-song.github.io/blog/2021/score/)] [[Github](https://github.com/ermongroup/ncsn)] \
12 Jul 2019 


### Classification


**Your Diffusion Model is Secretly a Zero-Shot Classifier** \
*Alexander C. Li, Mihir Prabhudesai, Shivam Duggal, Ellis Brown, Deepak Pathak* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.16203)] [[Project](https://diffusion-classifier.github.io/)] \
28 Mar 2023

**Diffusion Denoising Process for Perceptron Bias in Out-of-distribution Detection** \
*Luping Liu, Yi Ren, Xize Cheng, Zhou Zhao* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.11255)] [[Github](https://github.com/luping-liu/DiffOOD)] \
21 Nov 2022

**DiffusionDet: Diffusion Model for Object Detection** \
*Shoufa Chen, Peize Sun, Yibing Song, Ping Luo* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.09788)] [[Github](https://github.com/ShoufaChen/DiffusionDet)] \
17 Nov 2022

**Denoising Diffusion Models for Out-of-Distribution Detection** \
*Mark S. Graham, Walter H.L. Pinaya, Petru-Daniel Tudosiu, Parashkev Nachev, Sebastien Ourselin, M. Jorge Cardoso* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.07740)] [[Github](https://github.com/marksgraham/ddpm-ood)] \
14 Nov 2022

**From Points to Functions: Infinite-dimensional Representations in Diffusion Models** \
*Sarthak Mittal, Guillaume Lajoie, Stefan Bauer, Arash Mehrjou* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.13774)] [[Github](https://github.com/sarthmit/traj_drl)] \
25 Oct 2022


### Segmentation


**Multi-Level Global Context Cross Consistency Model for Semi-Supervised Ultrasound Image Segmentation with Diffusion Model** \
*Fenghe Tang, Jianrui Ding, Lingtao Wang, Min Xian, Chunping Ning* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09447)] [[Github](https://github.com/FengheTan9/Multi-Level-Global-Context-Cross-Consistency)] \
16 May 2023

**DiffuseExpand: Expanding dataset for 2D medical image segmentation using diffusion models** \
*Shitong Shao, Xiaohan Yuan, Zhen Huang, Ziming Qiu, Shuai Wang, Kevin Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.13416)] [[Github](https://anonymous.4open.science/r/DiffuseExpand/README.md)] \
26 Apr 2023

**Ambiguous Medical Image Segmentation using Diffusion Models** \
*Aimon Rahman, Jeya Maria Jose Valanarasu, Ilker Hacihaliloglu, Vishal M Patel* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2304.04745)] [[Github](https://github.com/aimansnigdha/Ambiguous-Medical-Image-Segmentation-using-Diffusion-Models)] \
10 Apr 2023

**DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models** \
*Weijia Wu, Yuzhong Zhao, Mike Zheng Shou, Hong Zhou, Chunhua Shen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11681)] [[Project](https://weijiawu.github.io/DiffusionMask/)] \
21 Mar 2023

**Diff-UNet: A Diffusion Embedded Network for Volumetric Segmentation** \
*Zhaohu Xing, Liang Wan, Huazhu Fu, Guang Yang, Lei Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10326)] [[Github](https://github.com/ge-xing/Diff-UNet)] \
18 Mar 2023

**Stochastic Segmentation with Conditional Categorical Diffusion Models** \
*Lukas Zbinden<sup>1</sup>, Lars Doorenbos<sup>1</sup>, Theodoros Pissas, Raphael Sznitman, Pablo Márquez-Neila* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08888)] [[Github](https://github.com/LarsDoorenbos/ccdm-stochastic-segmentation)] \
15 Mar 2023

**Importance of Aligning Training Strategy with Evaluation for Diffusion Models in 3D Multiclass Segmentation** \
*Yunguan Fu, Yiwen Li, Shaheer U. Saeed, Matthew J. Clarkson, Yipeng Hu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.06040)] [[Github](https://github.com/mathpluscode/ImgX-DiffSeg)] \
10 Mar 2023

**DiffusionInst: Diffusion Model for Instance Segmentation** \
*Zhangxuan Gu, Haoxing Chen, Zhuoer Xu, Jun Lan, Changhua Meng, Weiqiang Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.02773)] [[Github](https://github.com/chenhaoxing/DiffusionInst)] \
6 DEc 2022

**Remote Sensing Change Detection (Segmentation) using Denoising Diffusion Probabilistic Models** \
*Wele Gedara Chaminda Bandara, Nithin Gopalakrishnan Nair, Vishal M. Patel* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.11892)] [[Github](https://github.com/wgcban/ddpm-cd)] \
23 Jun 2022

**Label-Efficient Semantic Segmentation with Diffusion Models** \
*Dmitry Baranchuk, Ivan Rubachev, Andrey Voynov, Valentin Khrulkov, Artem Babenko* \
ICLR 2021. [[Paper](https://arxiv.org/abs/2112.03126)] [[Github](https://github.com/yandex-research/ddpm-segmentation)] \
6 Dec 2021


### Image Translation


**Null-text Guidance in Diffusion Models is Secretly a Cartoon-style Creator** \
*Jing Zhao, Heliang Zheng, Chaoyue Wang, Long Lan, Wanrong Huang, Wenjing Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.06710)] [[Project](https://nulltextforcartoon.github.io/)] [[Github](https://github.com/NullTextforCartoon/NullTextforCartoon)] \
11 May 2023

**DiffusionRig: Learning Personalized Priors for Facial Appearance Editing** \
*Zheng Ding, Xuaner Zhang, Zhihao Xia, Lars Jebe, Zhuowen Tu, Xiuming Zhang* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2304.06711)] [[Project](https://diffusionrig.github.io/)] [[Github](https://github.com/adobe-research/diffusion-rig)] \
13 Apr 2023

**Training-free Style Transfer Emerges from h-space in Diffusion models** \
*Jaeseok Jeong<sup>1</sup>, Mingi Kwon<sup>1</sup>, Youngjung Uh* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.15403)] [[Project](https://curryjung.github.io/DiffStyle/)] [[Github](https://github.com/curryjung/DiffStyle_official)] \
27 Mar 2023

**Pretraining is All You Need for Image-to-Image Translation** \
*Tengfei Wang, Ting Zhang, Bo Zhang, Hao Ouyang, Dong Chen, Qifeng Chen, Fang Wen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.12952)] [[Project](https://tengfei-wang.github.io/PITI/index.html)] [[Github](https://github.com/PITI-Synthesis/PITI)] \
25 May 2022


### Inverse Problems


**Exploiting Diffusion Prior for Real-World Image Super-Resolution** \
*Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin C.K. Chan, Chen Change Loy* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.07015)] [[Project](https://iceclear.github.io/projects/stablesr/)] [[Github](https://github.com/IceClear/StableSR)] \
11 May 2023

**Image Super-Resolution via Iterative Refinement**  \
*Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, Mohammad Norouzi* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2104.07636)] [[Project](https://iterative-refinement.github.io/)] [[Github](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)] \
15 Apr 2021


### Medical Imaging


**Multi-Level Global Context Cross Consistency Model for Semi-Supervised Ultrasound Image Segmentation with Diffusion Model** \
*Fenghe Tang, Jianrui Ding, Lingtao Wang, Min Xian, Chunping Ning* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09447)] [[Github](https://github.com/FengheTan9/Multi-Level-Global-Context-Cross-Consistency)] \
16 May 2023

**DiffuseExpand: Expanding dataset for 2D medical image segmentation using diffusion models** \
*Shitong Shao, Xiaohan Yuan, Zhen Huang, Ziming Qiu, Shuai Wang, Kevin Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.13416)] [[Github](https://anonymous.4open.science/r/DiffuseExpand/README.md)] \
26 Apr 2023

**Ambiguous Medical Image Segmentation using Diffusion Models** \
*Aimon Rahman, Jeya Maria Jose Valanarasu, Ilker Hacihaliloglu, Vishal M Patel* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2304.04745)] [[Github](https://github.com/aimansnigdha/Ambiguous-Medical-Image-Segmentation-using-Diffusion-Models)] \
10 Apr 2023

**Diff-UNet: A Diffusion Embedded Network for Volumetric Segmentation** \
*Zhaohu Xing, Liang Wan, Huazhu Fu, Guang Yang, Lei Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10326)] [[Github](https://github.com/ge-xing/Diff-UNet)] \
18 Mar 2023

**Class-Guided Image-to-Image Diffusion: Cell Painting from Brightfield Images with Class Labels** \
*Jan Oscar Cross-Zamirski, Praveen Anand, Guy Williams, Elizabeth Mouchet, Yinhai Wang, Carola-Bibiane Schönlieb* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08863)] [[Github](https://github.com/crosszamirski/guided-I2I)] \
15 Mar 2023


**Stochastic Segmentation with Conditional Categorical Diffusion Models** \
*Lukas Zbinden<sup>1</sup>, Lars Doorenbos<sup>1</sup>, Theodoros Pissas, Raphael Sznitman, Pablo Márquez-Neila* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08888)] [[Github](https://github.com/LarsDoorenbos/ccdm-stochastic-segmentation)] \
15 Mar 2023

**Importance of Aligning Training Strategy with Evaluation for Diffusion Models in 3D Multiclass Segmentation** \
*Yunguan Fu, Yiwen Li, Shaheer U. Saeed, Matthew J. Clarkson, Yipeng Hu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.06040)] [[Github](https://github.com/mathpluscode/ImgX-DiffSeg)] \
10 Mar 2023

**DDM2: Self-Supervised Diffusion MRI Denoising with Generative Diffusion Models** \
*Tiange Xiang, Mahmut Yurt, Ali B Syed, Kawin Setsompop, Akshay Chaudhari* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2302.03018)] [[Github](https://github.com/StanfordMIMI/DDM2)] \
6 Feb 2023

**Spot the fake lungs: Generating Synthetic Medical Images using Neural Diffusion Models** \
*Hazrat Ali, Shafaq Murad, Zubair Shah* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.00902)] [[Project](https://www.kaggle.com/datasets/hazrat/awesomelungs)] \
2 Nov 2022

**Multitask Brain Tumor Inpainting with Diffusion Models: A Methodological Report** \
*Pouria Rouzrokh<sup>1</sup>, Bardia Khosravi<sup>1</sup>, Shahriar Faghani, Mana Moassefi, Sanaz Vahdati, Bradley J. Erickson* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.12113)] [[Github](https://github.com/Mayo-Radiology-Informatics-Lab/MBTI)] \
21 Oct 2022

**Diffusion Deformable Model for 4D Temporal Medical Image Generation** \
*Boah Kim, Jong Chul Ye* \
MICCAI 2022. [[Paper](https://arxiv.org/abs/2206.13295)] [[Github](https://github.com/torchddm/ddm)] \
27 Jun 2022

**Diffusion Models for Medical Anomaly Detection** \
*Julia Wolleb, Florentin Bieder, Robin Sandkühler, Philippe C. Cattin* \
MICCAI 2022. [[Paper](https://arxiv.org/abs/2203.04306)] [[Github](https://github.com/JuliaWolleb/diffusion-anomaly)] \
8 Mar 2022

**Towards performant and reliable undersampled MR reconstruction via diffusion model sampling** \
*Cheng Peng, Pengfei Guo, S. Kevin Zhou, Vishal Patel, Rama Chellappa* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2203.04292)] [[Github](https://github.com/cpeng93/diffuserecon)] \
8 Mar 2022

**Measurement-conditioned Denoising Diffusion Probabilistic Model for Under-sampled Medical Image Reconstruction** \
*Yutong Xie, Quanzheng Li* \
MICCAI 2022. [[Paper](https://arxiv.org/abs/2203.03623)] [[Github](https://github.com/Theodore-PKU/MC-DDPM)] \
5 Mar 2022

**Unsupervised Denoising of Retinal OCT with Diffusion Probabilistic Model** \
*Dewei Hu, Yuankai K. Tao, Ipek Oguz* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2201.11760)] [[Github](https://github.com/DeweiHu/OCT_DDPM)] \
27 Jan 2022

**Score-based diffusion models for accelerated MRI** \
*Hyungjin Chung, Jong chul Ye* \
MIA 2021. [[Paper](https://arxiv.org/abs/2110.05243)] [[Github](https://github.com/HJ-harry/score-MRI)] \
8 Oct 2021


### Multi-modal Learning


**Null-text Guidance in Diffusion Models is Secretly a Cartoon-style Creator** \
*Jing Zhao, Heliang Zheng, Chaoyue Wang, Long Lan, Wanrong Huang, Wenjing Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.06710)] [[Project](https://nulltextforcartoon.github.io/)] [[Github](https://github.com/NullTextforCartoon/NullTextforCartoon)] \
11 May 2023

**In-Context Learning Unlocked for Diffusion Models** \
*Zhendong Wang, Yifan Jiang, Yadong Lu, Yelong Shen, Pengcheng He, Weizhu Chen, Zhangyang Wang, Mingyuan Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.01115)] [[Project](https://zhendong-wang.github.io/prompt-diffusion.github.io/)] [[Github](https://github.com/Zhendong-Wang/Prompt-Diffusion)] \
1 May 2023

**Expressive Text-to-Image Generation with Rich Text** \
*Songwei Ge, Taesung Park, Jun-Yan Zhu, Jia-Bin Huang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06720)] [[Project](https://rich-text-to-image.github.io/)] [[Github](https://github.com/SongweiGe/rich-text-to-image)] \
13 Apr 2023

**ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model** \
*Mingyuan Zhang, Xinying Guo, Liang Pan, Zhongang Cai, Fangzhou Hong, Huirong Li, Lei Yang, Ziwei Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01116)] [[Project](https://mingyuan-zhang.github.io/projects/ReMoDiffuse.html)] [[Github](https://github.com/mingyuan-zhang/ReMoDiffuse)] \
3 Apr 2023

**AvatarCraft: Transforming Text into Neural Human Avatars with Parameterized Shape and Pose Control** \
*Ruixiang Jiang, Can Wang, Jingbo Zhang, Menglei Chai, Mingming He, Dongdong Chen, Jing Liao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17606)] [[Project](https://avatar-craft.github.io/)] [[Github](https://github.com/songrise/avatarcraft)] \
30 Mar 2023

**ReVersion: Diffusion-Based Relation Inversion from Images** \
*Ziqi Huang<sup>1</sup>, Tianxing Wu<sup>1</sup>, Yuming Jiang, Kelvin C.K. Chan, Ziwei Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13495)] [[Project](https://ziqihuangg.github.io/projects/reversion.html)] [[Github](https://github.com/ziqihuangg/ReVersion)]
23 Mar 2023

**Ablating Concepts in Text-to-Image Diffusion Models** \
*Nupur Kumari, Bingliang Zhang, Sheng-Yu Wang, Eli Shechtman, Richard Zhang, Jun-Yan Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13516)] [[Project](https://www.cs.cmu.edu/~concept-ablation/)] [[Github](https://github.com/nupurkmr9/concept-ablation)] \
23 Mar 2023

**MagicFusion: Boosting Text-to-Image Generation Performance by Fusing Diffusion Models** \
*Jing Zhao, Heliang Zheng, Chaoyue Wang, Long Lan, Wenjing Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13126)] [[Project](https://magicfusion.github.io/)] [[Github](https://github.com/MagicFusion/MagicFusion.github.io)] \
23 Mar 2023

**FateZero: Fusing Attentions for Zero-shot Text-based Video Editing** \
*Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, Qifeng Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09535)] [[Project](https://fate-zero-edit.github.io/)] [[Github](https://github.com/ChenyangQiQi/FateZero)] \
16 Mar 2023

**Editing Implicit Assumptions in Text-to-Image Diffusion Models** \
*Hadas Orgad<sup>1</sup>, Bahjat Kawar<sup>1</sup>, Yonatan Belinkov* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08084)] [[Project](https://time-diffusion.github.io/)] [[Github](https://github.com/bahjat-kawar/time-diffusion)] \
14 Mar 2023

**Erasing Concepts from Diffusion Models** \
*Rohit Gandikota<sup>1</sup>, Joanna Materzynska<sup>1</sup>, Jaden Fiotto-Kaufman, David Bau* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.07345)] [[Project](https://erasing.baulab.info/)] [[Github](https://github.com/rohitgandikota/erasing)] \
13 Mar 2023

**MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation** \
*Omer Bar-Tal<sup>1</sup>, Lior Yariv<sup>1</sup>, Yaron Lipman, Tali Dekel* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.08113)] [Project](https://multidiffusion.github.io/)] [[Github](https://github.com/omerbt/MultiDiffusion)] \
16 Feb 2023

**TEXTure: Text-Guided Texturing of 3D Shapes** \
*Elad Richardson<sup>1</sup>, Gal Metzer<sup>1</sup>, Yuval Alaluf, Raja Giryes, Daniel Cohen-Or* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.01721)] [[Project](https://texturepaper.github.io/TEXTurePaper/)] [[Github](https://github.com/TEXTurePaper/TEXTurePaper)] \
3 Feb 2023

**Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models** \
*Hila Chefer<sup>1</sup>, Yuval Alaluf<sup>1</sup>, Yael Vinker, Lior Wolf, Daniel Cohen-Or* \
SIGGRAPH 2023. [[Paper](https://arxiv.org/abs/2301.13826)] [[Project](https://attendandexcite.github.io/Attend-and-Excite/)] [[Github](https://github.com/AttendAndExcite/Attend-and-Excite)] \
31 Jan 2023

**Speech Driven Video Editing via an Audio-Conditioned Diffusion Model** \
*Dan Bigioi, Shubhajit Basak, Hugh Jordan, Rachel McDonnell, Peter Corcoran* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.04474)] [[Project](https://danbigioi.github.io/DiffusionVideoEditing/)] [[Github](https://github.com/DanBigioi/DiffusionVideoEditing)] \
10 Jan 2023

**Optimizing Prompts for Text-to-Image Generation** \
*Yaru Hao<sup>1</sup>, Zewen Chi<sup>1</sup>, Li Dong, Furu Wei* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09611)] [[Project](https://huggingface.co/spaces/microsoft/Promptist)] [[Github](https://github.com/microsoft/LMOps/tree/main/promptist)] \
19 Dec 2022

**SINE: SINgle Image Editing with Text-to-Image Diffusion Models** \
*Zhixing Zhang, Ligong Han, Arnab Ghosh, Dimitris Metaxas, Jian Ren* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.04489)] [[Project](https://zhang-zx.github.io/SINE/)] [[Github](https://github.com/zhang-zx/SINE)] \
8 Dec 2022

**Diffusion-SDF: Text-to-Shape via Voxelized Diffusion** \
*Muheng Li, Yueqi Duan, Jie Zhou, Jiwen Lu* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.03293)] [[Project](https://ttlmh.github.io/DiffusionSDF/)] [[Github](https://github.com/ttlmh/Diffusion-SDF)] \
6 Dec 2022

**Diffusion Video Autoencoders: Toward Temporally Consistent Face Video Editing via Disentangled Video Encoding** \
*Gyeongman Kim, Hajin Shim, Hyunsu Kim, Yunjey Choi, Junho Kim, Eunho Yang* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.02802)] [[Project](https://diff-video-ae.github.io/)] [[Github](https://github.com/man805/Diffusion-Video-Autoencoders)] \
6 Dec 2022

**InstructPix2Pix: Learning to Follow Image Editing Instructions** \
*Tim Brooks, Aleksander Holynski, Alexei A. Efros* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2211.09800)] [[Project](https://www.timothybrooks.com/instruct-pix2pix)] [[Github](https://github.com/timothybrooks/instruct-pix2pix)] \
17 Nov 2022

**DiffusionDB: A Large-scale Prompt Gallery Dataset for Text-to-Image Generative Models** \
*Zijie J. Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, Duen Horng Chau* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.14896)] [[Project](https://poloclub.github.io/diffusiondb/)] [[Github](https://github.com/poloclub/diffusiondb)] \
26 Oct 2022

**DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation** \
*Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, Kfir Aberman* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2208.12242)] [[Project](https://dreambooth.github.io/)] [[Github](https://github.com/Victarry/stable-dreambooth)] \
25 Aug 2022

**Blended Latent Diffusion** \
*Omri Avrahami, Ohad Fried, Dani Lischinski* \
ACM 2022. [[Paper](https://arxiv.org/abs/2206.02779)] [[Project](https://omriavrahami.com/blended-latent-diffusion-page/)] [[Github](https://github.com/omriav/blended-latent-diffusion)] \
6 Jun 2022

**Compositional Visual Generation with Composable Diffusion Models** \
*Nan Liu<sup>1</sup>, Shuang Li<sup>1</sup>, Yilun Du<sup>1</sup>, Antonio Torralba, Joshua B. Tenenbaum* \
ECCV 2022. [[Paper](https://arxiv.org/abs/2206.01714)] [[Project](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/)] [[Github](https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch)] \
3 Jun 2022

**Blended Diffusion for Text-driven Editing of Natural Images** \
*Omri Avrahami, Dani Lischinski, Ohad Fried* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2111.14818)] [[Project](https://omriavrahami.com/blended-diffusion-page/)] [[Github](https://github.com/omriav/blended-diffusion)] \
29 Nov 2021


### 3D Vision


**Nerfbusters: Removing Ghostly Artifacts from Casually Captured NeRFs** \
*Frederik Warburg, Ethan Weber, Matthew Tancik, Aleksander Holynski, Angjoo Kanazawa* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.10532)] [[Project](https://ethanweber.me/nerfbusters/)] [[Github](https://github.com/ethanweber/nerfbusters)] \
20 Apr 2023

**ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model** \
*Mingyuan Zhang, Xinying Guo, Liang Pan, Zhongang Cai, Fangzhou Hong, Huirong Li, Lei Yang, Ziwei Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01116)] [[Project](https://mingyuan-zhang.github.io/projects/ReMoDiffuse.html)] [[Github](https://github.com/mingyuan-zhang/ReMoDiffuse)] \
3 Apr 2023

**AvatarCraft: Transforming Text into Neural Human Avatars with Parameterized Shape and Pose Control** \
*Ruixiang Jiang, Can Wang, Jingbo Zhang, Menglei Chai, Mingming He, Dongdong Chen, Jing Liao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17606)] [[Project](https://avatar-craft.github.io/)] [[Github](https://github.com/songrise/avatarcraft)] \
30 Mar 2023

**Instruct 3D-to-3D: Text Instruction Guided 3D-to-3D conversion** \
*Hiromichi Kamata, Yuiko Sakuma, Akio Hayakawa, Masato Ishii, Takuya Narihira* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.15780)] [[Project](https://sony.github.io/Instruct3Dto3D-doc/)] [[Github](https://sony.github.io/Instruct3Dto3D-doc/)] \
28 Mar 2023

**Novel View Synthesis of Humans using Differentiable Rendering** \
*Guillaume Rochette, Chris Russell, Richard Bowden* \
IEEE T-BIOM 2023. [[Paper](https://arxiv.org/abs/2303.15880)] [[Github](https://github.com/GuillaumeRochette/HumanViewSynthesis)] \
28 Mar 2023

**Make-It-3D: High-Fidelity 3D Creation from A Single Image with Diffusion Prior** \
*Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, Dong Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.14184)] [[Project](https://make-it-3d.github.io/)] [[Github](https://make-it-3d.github.io/)] \
24 Mar 2023

**Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation** \
*Rui Chen, Yongwei Chen, Ningxin Jiao, Kui Jia* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13873)] [[Project](https://fantasia3d.github.io/)] [[Github](https://github.com/Gorilla-Lab-SCUT/Fantasia3D)] \
24 Mar 2023

**Zero-1-to-3: Zero-shot One Image to 3D Object** \
*Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, Carl Vondrick* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11328)] [[Project](https://zero123.cs.columbia.edu/)] [[Github](https://github.com/cvlab-columbia/zero123)] \
20 Mar 2023

**MeshDiffusion: Score-based Generative 3D Mesh Modeling** \
*Zhen Liu, Yao Feng, Michael J. Black, Derek Nowrouzezahrai, Liam Paull, Weiyang Liu* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2303.08133)] [[Project](https://meshdiffusion.github.io/)] [[Github](https://github.com/lzzcd001/MeshDiffusion/)] \
14 Mar 2023

**Can We Use Diffusion Probabilistic Models for 3D Motion Prediction?** \
*Hyemin Ahn, Esteve Valls Mascaro, Dongheui Lee* \
ICRA 2023. [[Paper](https://arxiv.org/abs/2302.14503)] [[Project](https://sites.google.com/view/diffusion-motion-prediction)] [[Github](https://github.com/cotton-ahn/diffusion-motion-prediction)] \
28 Feb 2023

**SinMDM: Single Motion Diffusion** \
*Sigal Raab<sup>1</sup>, Inbal Leibovitch<sup>1</sup>, Guy Tevet, Moab Arar, Amit H. Bermano, Daniel Cohen-Or* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.05905)] [[Project](https://sinmdm.github.io/SinMDM-page/)] [[Github](https://github.com/SinMDM/SinMDM)] \
12 Feb 2023

**HumanMAC: Masked Motion Completion for Human Motion Prediction** \
*Ling-Hao Chen, Jiawei Zhang, Yewen Li, Yiren Pang, Xiaobo Xia, Tongliang Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03665)] [[Project](https://lhchen.top/Human-MAC/)] [[Github](https://github.com/LinghaoChan/HumanMAC)] \
7 Feb 2023

**TEXTure: Text-Guided Texturing of 3D Shapes** \
*Elad Richardson<sup>1</sup>, Gal Metzer<sup>1</sup>, Yuval Alaluf, Raja Giryes, Daniel Cohen-Or* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.01721)] [[Project](https://texturepaper.github.io/TEXTurePaper/)] [[Github](https://github.com/TEXTurePaper/TEXTurePaper)] \
3 Feb 2023

**Diffusion-based Generation, Optimization, and Planning in 3D Scenes** \
*Siyuan Huang<sup>1</sup>, Zan Wang<sup>1</sup>, Puhao Li, Baoxiong Jia, Tengyu Liu, Yixin Zhu, Wei Liang, Song-Chun Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.06015)] [[Project](https://scenediffuser.github.io/)] [[Github](https://github.com/scenediffuser/Scene-Diffuser)] \
15 Jan 2023

**Generative Scene Synthesis via Incremental View Inpainting using RGBD Diffusion Models** \
*Jiabao Lei, Jiapeng Tang, Kui Jia* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.05993)] [[Project](https://jblei.site/project-pages/rgbd-diffusion.html)] [[Github](https://github.com/Karbo123/RGBD-Diffusion)] \
12 Dec 2022

**Executing your Commands via Motion Diffusion in Latent Space** \
*Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, Jingyi Yu, Gang Yu* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.04048)] [[Project](https://chenxin.tech/mld/)] [[Github](https://github.com/ChenFengYe/motion-latent-diffusion)] \
8 Dec 2022

**SparseFusion: Distilling View-conditioned Diffusion for 3D Reconstruction** \
*Zhizhuo Zhou, Shubham Tulsiani* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.00792)] [[Project](https://sparsefusion.github.io/)] [[Github](https://sparsefusion.github.io/)] \
1 Dec 2022

**NeuralLift-360: Lifting An In-the-wild 2D Photo to A 3D Object with 360° Views** \
*Dejia Xu, Yifan Jiang, Peihao Wang, Zhiwen Fan, Yi Wang, Zhangyang Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.16431)] [[Project](https://vita-group.github.io/NeuralLift-360/)] [[Github](https://github.com/VITA-Group/NeuralLift-360)] \
29 Nov 2022

**UDE: A Unified Driving Engine for Human Motion Generation** \
*Zixiang Zhou, Baoyuan Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.16016)] [[Project](https://zixiangzhou916.github.io/UDE/)] [[Github](https://github.com/zixiangzhou916/UDE/)] \
29 Nov 2022

**DiffuStereo: High Quality Human Reconstruction via Diffusion-based Stereo Using Sparse Cameras** \
*Ruizhi Shao, Zerong Zheng, Hongwen Zhang, Jingxiang Sun, Yebin Liu* \
ECCV 2022. [[Paper](https://arxiv.org/abs/2207.08000)] [[Project](http://liuyebin.com/diffustereo/diffustereo.html)] [[Github](https://github.com/DSaurus/DiffuStereo)] \
16 Jul 2022

### Adversarial Attack


**Diffusion Models for Adversarial Purification** \
*Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, Anima Anandkumar* \
ICML 2022. [[Paper](https://arxiv.org/abs/2205.07460)] [[Project](https://diffpure.github.io/)] [[Github](https://github.com/NVlabs/DiffPure)] \
16 May 2022



## Audio



### Generation


**FastDiff: A Fast Conditional Diffusion Model for High-Quality Speech Synthesis** \
*Rongjie Huang<sup>1</sup>, Max W. Y. Lam<sup>1</sup>, Jun Wang, Dan Su, Dong Yu, Yi Ren, Zhou Zhao* \
IJCAI 2022. [[Paper](https://arxiv.org/abs/2204.09934)] [[Project](https://fastdiff.github.io/)] [[Github](https://github.com/Rongjiehuang/FastDiff)] \
21 Apr 2022

**DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism** \
*Jinglin Liu<sup>1</sup>, Chengxi Li<sup>1</sup>, Yi Ren<sup>1</sup>, Feiyang Chen, Peng Liu, Zhou Zhao* \
AAAI 2022. [[Paper](https://arxiv.org/abs/2105.02446)] [[Project](https://diffsinger.github.io/)] [[Github](https://github.com/keonlee9420/DiffSinger)] \
6 May 2021

**WaveGrad: Estimating Gradients for Waveform Generation** \
*Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, William Chan*\
ICLR 2021. [[Paper](https://arxiv.org/abs/2009.00713)] [[Project](https://wavegrad.github.io/)] [[Github](https://github.com/ivanvovk/WaveGrad)] \
2 Sep 2020 


### Conversion

**DiffSVC: A Diffusion Probabilistic Model for Singing Voice Conversion**  \
*Songxiang Liu<sup>1</sup>, Yuewen Cao<sup>1</sup>, Dan Su, Helen Meng* \
IEEE 2021. [[Paper](https://arxiv.org/abs/2105.13871)] [[Github](https://github.com/liusongxiang/diffsvc)] \
28 May 2021

**Diffusion-Based Voice Conversion with Fast Maximum Likelihood Sampling Scheme** \
*Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, Mikhail Kudinov, Jiansheng Wei* \
ICLR 2022. [[Paper](https://arxiv.org/abs/2109.13821)] [[Project](https://diffvc-fast-ml-solver.github.io/)] \
28 Sep 2021

### Enhancement


**Analysing Diffusion-based Generative Approaches versus Discriminative Approaches for Speech Restoration** \
*Jean-Marie Lemercier, Julius Richter, Simon Welker, Timo Gerkmann* \
Interspeech 2022. [[Paper](https://arxiv.org/abs/2211.02397)] [[Project](https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/sgmse-multitask.html)] [[Github](https://github.com/sp-uhh/sgmse)] \
4 Nov 2022

**Conditioning and Sampling in Variational Diffusion Models for Speech Super-resolution** \
*Chin-Yun Yu, Sung-Lin Yeh, György Fazekas, Hao Tang* \
ICASSP 2023. [[Paper](https://arxiv.org/abs/2210.15793)] [[Project](https://yoyololicon.github.io/diffwave-sr/)] [[Github](https://github.com/yoyololicon/diffwave-sr)] \
27 Oct 2022

**Speech Enhancement and Dereverberation with Diffusion-based Generative Models** \
*Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, Timo Gerkmann* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.05830)] [[Project](https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/sgmse)] [[Github](https://github.com/sp-uhh/sgmse)] \
11 Aug 2022

**NU-Wave: A Diffusion Probabilistic Model for Neural Audio Upsampling**  \
*Junhyeok Lee, Seungu Han* \
Interspeech 2021. [[Paper](https://arxiv.org/abs/2104.02321)] [[Project](https://mindslab-ai.github.io/nuwave/)] [[Github](https://github.com/mindslab-ai/nuwave)] \
6 Apr 2021


### Separation


**Multi-Source Diffusion Models for Simultaneous Music Generation and Separation** \
*Giorgio Mariani<sup>1</sup>, Irene Tallini<sup>1</sup>, Emilian Postolache<sup>1</sup>, Michele Mancusi<sup>1</sup>, Luca Cosmo, Emanuele Rodolà* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.02257)] [[Project](https://gladia-research-group.github.io/multi-source-diffusion-models/)] \
4 Feb 2023

**StoRM: A Diffusion-based Stochastic Regeneration Model for Speech Enhancement and Dereverberation** \
*Jean-Marie Lemercier, Julius Richter, Simon Welker, Timo Gerkmann* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.11851)] [[Github](https://github.com/sp-uhh/storm)] \
22 Dec 2022


### Text-to-Speech


**Text-to-Audio Generation using Instruction-Tuned LLM and Latent Diffusion Model** \
*Deepanway Ghosal, Navonil Majumder, Ambuj Mehrish, Soujanya Poria* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.13731)] [[Project](https://tango-web.github.io/)] [[Github](https://github.com/declare-lab/tango)] \
24 Apr 2023

**AudioLDM: Text-to-Audio Generation with Latent Diffusion Models** \
*Haohe Liu<sup>1</sup>, Zehua Chen<sup>1</sup>, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, Mark D. Plumbley* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.12503)] [[Project](https://audioldm.github.io/)] [[Github](https://github.com/haoheliu/AudioLDM)] \
29 Jan 2023

**EdiTTS: Score-based Editing for Controllable Text-to-Speech** \
*Jaesung Tae<sup>1</sup>, Hyeongju Kim<sup>1</sup>, Taesu Kim* \
Interspeech 2022. [[Paper](https://arxiv.org/abs/2110.02584)] [[Project](https://editts.github.io/)] [[Github](https://github.com/neosapience/EdiTTS)] \
6 Oct 2021

**WaveGrad 2: Iterative Refinement for Text-to-Speech Synthesis** \
*Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, Najim Dehak, William Chan* \
Interspeech 2021. [[Paper](https://arxiv.org/abs/2106.09660)] [[Project](https://mindslab-ai.github.io/wavegrad2/)] [[Github](https://github.com/keonlee9420/WaveGrad2)] [[Github2](https://github.com/mindslab-ai/wavegrad2)] \
17 Jun 2021 

**Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech** \
*Vadim Popov<sup>1</sup>, Ivan Vovk<sup>1</sup>, Vladimir Gogoryan, Tasnima Sadekova, Mikhail Kudinov* \
ICML 2021. [[Paper](https://arxiv.org/abs/2105.06337)] [[Project](https://grad-tts.github.io/)] [[Github](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS)] \
13 May 2021

**DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism** \
*Jinglin Liu<sup>1</sup>, Chengxi Li<sup>1</sup>, Yi Ren<sup>1</sup>, Feiyang Chen, Peng Liu, Zhou Zhao* \
AAAI 2022. [[Paper](https://arxiv.org/abs/2105.02446)] [[Project](https://diffsinger.github.io/)] [[Github](https://github.com/keonlee9420/DiffSinger)] \
6 May 2021



## Natural Language



**Can Diffusion Model Achieve Better Performance in Text Generation? Bridging the Gap between Training and Inference!** \
*Zecheng Tang<sup>1</sup>, Pinzheng Wang<sup>1</sup>, Keyan Zhou, Juntao Li, Ziqiang Cao, Min Zhang* \
ACL 2023. [[Paper](https://arxiv.org/abs/2305.04465)] [[Github](https://github.com/CODINNLG/Bridge_Gap_Diffusion)] \
8 May 2023

**DocDiff: Document Enhancement via Residual Diffusion Models** \
*Zongyuan Yang, Baolin Liu, Yongping Xiong, Lan Yi, Guibin Wu, Xiaojun Tang, Ziqi Liu, Junjie Zhou, Xing Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03892)] [[Github](https://github.com/Royalvice/DocDiff)] \
6 May 2023

**A Cheaper and Better Diffusion Language Model with Soft-Masked Noise** \
*Jiaao Chen, Aston Zhang, Mu Li, Alex Smola, Diyi Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.04746)] [[Github](https://github.com/amazon-science/masked-diffusion-lm)] \
10 Apr 2023

**A Reparameterized Discrete Diffusion Model for Text Generation** \
*Lin Zheng, Jianbo Yuan, Lei Yu, Lingpeng Kong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.05737)] [[Github](https://github.com/HKUNLP/reparam-discrete-diffusion)] \
11 Feb 2023

**DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models** \
*Zhengfu He<sup>1</sup>, Tianxiang Sun<sup>1</sup>, Kuanning Wang, Xuanjing Huang, Xipeng Qiu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.15029)] [[Github](https://github.com/Hzfinfdu/Diffusion-BERT)] \
28 Nov 2022

**SSD-LM: Semi-autoregressive Simplex-based Diffusion Language Model for Text Generation and Modular Control** \
*Xiaochuang Han, Sachin Kumar, Yulia Tsvetkov* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.17432)] [[Github](https://github.com/xhan77/ssd-lm)] \
31 Oct 2022

**Latent Diffusion Energy-Based Model for Interpretable Text Modeling** \
*Peiyu Yu, Sirui Xie, Xiaojian Ma, Baoxiong Jia, Bo Pang, Ruigi Gao, Yixin Zhu, Song-Chun Zhu, Ying Nian Wu* \
ICML 2022. [[Paper](https://arxiv.org/abs/2206.05895)] [[Github](https://github.com/yuPeiyu98/LDEBM)] \
13 Jun 2022

**Diffusion-LM Improves Controllable Text Generation** \
*Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, Tatsunori B. Hashimoto* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2205.14217)] [[Github](https://github.com/XiangLi1999/Diffusion-LM)] \
27 May 2022



## Tabular and Time Series



### Generation


**EHRDiff: Exploring Realistic EHR Synthesis with Diffusion Models** \
*Hongyi Yuan<sup>1</sup>, Songchi Zhou<sup>1</sup>, Sheng Yu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.05656)] [[Github](https://github.com/sczzz3/ehrdiff)] \
10 Mar 2023

**TabDDPM: Modelling Tabular Data with Diffusion Models** \
*Akim Kotelnikov, Dmitry Baranchuk, Ivan Rubachev, Artem Babenko* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.15421)] [[Github](https://github.com/rotot0/tab-ddpm)] \
30 Sep 2022


### Forecasting


**TDSTF: Transformer-based Diffusion probabilistic model for Sparse Time series Forecasting** \
*Ping Chang, Huayu Li, Stuart F. Quan, Janet Roveda, Ao Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.06625)] [[Github](https://github.com/PingChang818/TDSTF)] \
16 Jan 2023

**Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models** \
*Juan Miguel Lopez Alcaraz, Nils Strodthoff* \
TMLR 2022. [[Paper](https://arxiv.org/abs/2208.09399)] [[Github](https://github.com/AI4HealthUOL/SSSD)] \
19 Aug 2022

**Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting** \
*Kashif Rasul, Calvin Seward, Ingmar Schuster, Roland Vollgraf* \
ICLR 2021. [[Paper](https://arxiv.org/abs/2101.12072)] [[Github](https://github.com/zalandoresearch/pytorch-ts)] \
2 Feb 2021 


### Imputation


**PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation** \
*Mingzhe Liu, Han Huang, Hao Feng, Leilei Sun, Bowen Du, Yanjie Fu* \
ICDE 2023. [[Paper](https://arxiv.org/abs/2302.09746)] [[Github](https://github.com/LMZZML/PriSTI)] \
20 Feb 2023

**Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models** \
*Juan Miguel Lopez Alcaraz, Nils Strodthoff* \
TMLR 2022. [[Paper](https://arxiv.org/abs/2208.09399)] [[Github](https://github.com/AI4HealthUOL/SSSD)] \
19 Aug 2022

**CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation** \
*Yusuke Tashiro, Jiaming Song, Yang Song, Stefano Ermon* \
NeurIPS 2021. [[Paper](https://arxiv.org/abs/2107.03502)] [[Github](https://github.com/ermongroup/csdi)]\
7 Jul 2021 



## Graph



### Generation


**Permutation Invariant Graph Generation via Score-Based Generative Modeling** \
*Chenhao Niu, Yang Song, Jiaming Song, Shengjia Zhao, Aditya Grover, Stefano Ermon* \
AISTATS 2021. [[Paper](https://arxiv.org/abs/2003.00638)] [[Github](https://github.com/ermongroup/GraphScoreMatching)] \
2 Mar 2020


### Molecular and Material Generation


**Protein Structure and Sequence Generation with Equivariant Denoising Diffusion Probabilistic Models** \
*Namrata Anand, Tudor Achim* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.15019)] [[Project](https://nanand2.github.io/proteins/)] [[Github](https://github.com/lucidrains/ddpm-ipa-protein-generation)] \
26 May 2022


## Theory


**Information-Theoretic Diffusion** \
*Xianghao Kong, Rob Brekelmans, Greg Ver Steeg* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2302.03792)] [[Github](https://github.com/gregversteeg/InfoDiffusionSimple)] \
7 Feb 2023

**Conditional Flow Matching: Simulation-Free Dynamic Optimal Transport** \
*Alexander Tong, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Kilian Fatras, Guy Wolf, Yoshua Bengio* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.00482)] [[Github](https://github.com/atong01/conditional-flow-matching)] \
1 Feb 2023

**Score-based Generative Modeling Secretly Minimizes the Wasserstein Distance** \
*Dohyun Kwon, Ying Fan, Kangwook Lee* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2212.06359)] [[Github](https://github.com/UW-Madison-Lee-Lab/score-wasserstein)] \
13 Dec 2022

**Diffusion Models for Causal Discovery via Topological Ordering** \
*Pedro Sanchez, Xiao Liu, Alison Q O'Neil, Sotirios A. Tsaftaris* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2210.06201)] [[Github](https://github.com/vios-s/DiffAN)] \
12 Oct 2022

**Theory and Algorithms for Diffusion Processes on Riemannian Manifolds** \
*Bowen Jing, Gabriele Corso, Jeffrey Chang, Regina Barzilay, Tommi Jaakkola* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.01729)] [[Github](https://github.com/gcorso/torsional-diffusion)] \
1 Jun 2022

**Bayesian Learning via Stochastic Gradient Langevin Dynamics** \
*Max Welling, Yee Whye Teh* \
ICML 2011. [[Paper](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)] [[Github](https://github.com/JavierAntoran/Bayesian-Neural-Networks#stochastic-gradient-langevin-dynamics-sgld)] \
20 Apr 2011

## Applications

**Dirichlet Diffusion Score Model for Biological Sequence Generation** \
*Pavel Avdeyev, Chenlai Shi, Yuhao Tan, Kseniia Dudnyk, Jian Zhou* \
ICML 2023 [[Paper](https://arxiv.org/abs/2305.10699)] [[Github](https://github.com/jzhoulab/ddsm)] \
18 May 2023

**GETMusic: Generating Any Music Tracks with a Unified Representation and Diffusion Framework** \
*Ang Lv, Xu Tan, Peiling Lu, Wei Ye, Shikun Zhang, Jiang Bian, Rui Yan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10841)] [[Github](https://github.com/microsoft/muzic)] \
18 May 2023

**Discrete Diffusion Probabilistic Models for Symbolic Music Generation** \
*Matthias Plasser, Silvan Peter, Gerhard Widmer* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09489)] [[Github](https://github.com/plassma/symbolic-music-discrete-diffusion)] \
16 May 2023
