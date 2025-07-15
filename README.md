# SoK: Gradient Inversion Attacks in Federated Learning
[![arXiv](https://img.shields.io/badge/usenix-paper-8f1d05.svg)](https://www.usenix.org/conference/usenixsecurity25/presentation/carletti)
[![license](https://img.shields.io/github/license/gparrella12/sok_gias_fl)](./LICENSE)



This repository hosts and maintains up-to-date systematization of Gradient Inversion Attacks and defenses in Federated Learning, following the categorization and threat model taxonomy introduced in our paper.

The tables are structured to provide a clear, comparative overview of the state-of-the-art, including:
- **Attack techniques** and their properties
- **Defense mechanisms** and their effectiveness
- **Threat models** with real-world applicability

Our goal is to offer a living resource for researchers and practitioners, reflecting the latest developments and evaluations in the field.
Tables will be updated regularly as new works and insights emerge.

Feel free to open issues or pull requests to suggest updates or corrections.

For further information, please contact: gparrella@unisa.it



## Threat Models
| **ID** | **Threat Model**             | **Model Updates** | **Basic Knowledge** | **Training Details** | **Surrogate Data** | **Client Data Distribution** | **Active Manipulation** | **Client Selection** | **Real-World Applicability** |
|--------|------------------------------|:-----------------:|:-------------------:|:--------------------:|:------------------:|:----------------------------:|:-----------------------:|:--------------------:|:----------------------------:|
| A      | Eavesdropper                 | ✔️                | ✗                   | ✗                    | ✗                  | ✗                            | ✗                       | ✗                    | ★★★                        |
| B      | Informed Eavesdropper        | ✔️                | ✔️                  | ✗                    | ✗                  | ✗                            | ✗                       | ✗                    | ★★★                        |
| C      | Parameter-Aware Eavesdropper | ✔️                | ✔️                  | ✔️                   | ✗                  | ✗                            | ✗                       | ✗                    | ★★★                        |
| D      | Data-Enhanced Eavesdropper   | ✔️                | ✔️                  | ✔️                   | ✔️                 | ✗                            | ✗                       | ✗                    | ★★☆                        |
| E      | Statistical-Informed Eavesdropper | ✔️           | ✔️                  | ✔️                   | ✔️                 | ✔️                           | ✗                       | ✗                    | ★★☆                        |
| F      | Active Manipulator           | ✔️                | ✔️                  | ✔️                   | ✗                  | ✗                            | ✔️                      | ✗                    | ★☆☆                        |
| G      | Data-Enhanced Manipulator    | ✔️                | ✔️                  | ✔️                   | ✔️                 | ✗                            | ✔️                      | ✗                    | ★☆☆                        |
| H      | Active Client Manipulator    | ✔️                | ✔️                  | ✔️                   | ✔️                 | ✗                            | ✔️                      | ✔️                   | ★☆☆                        |

**Legend for Applicability:**  
★★★ = highly applicable and less detectable  
★★☆ = potentially applicable (depends on specific configuration)  
★☆☆ = less applicable and more detectable

## Gradient Inversion Attacks

| **Category**         | **Technique**        | **Work**                                                                                          | **Threat Model** | **Learning Algorithm** | **Image Resolution** | **Batch Size** | **Shared Model** | **Learning Task**              | **Label <br> Recovery** | **Open Source**                                                                 |
|----------------------|----------------------|---------------------------------------------------------------------------------------------------------------|------------------|-----------------------|---------------------|---------------|------------------|-------------------------------|--------------------|---------------------------------------------------------------------------------|
| Optimization-based   | Basic Optimization   | Zhu et al. (2019) [1]                                           | A                | FedSGD                | 64×64               | 8             | LeNet            | Object and Face Classification | ★                  | [link](https://github.com/mit-han-lab/dlg)                                       |
|                      |                      | Zhao et al. (2020) [2]                              | A                | FedSGD                | 32×32               | 1             | LeNet            | Object and Face Classification | 🆕                 | [link](https://github.com/PatrickZH/Improved-Deep-Leakage-from-Gradients)         |
|                      |                      | Geiping et al. (2020) [3] | B                | FedSGD                | 32×32               | 100           | ResNets          | Object Classification          | ◦                  | [link](https://github.com/JonasGeiping/invertinggradients)                        |
|                      |                      | Yin et al. (2021) [4]             | E                | FedSGD                | 224×224             | 48            | ResNet-50        | Object Classification          | 🆕                 | ✗                                                                               |
|                      |                      | Hatamizadeh et al. (2022) [5]                | E                | FedSGD                | 224×224             | 30            | ViT              | Object and Face Classification | [4]                | ✗                                                                               |
|                      |                      | Dimitrov et al. (2022) [6]                                  | C                | FedAVG                | 32×32               | 10×5          | CNNs             | Object Classification          | [link](https://arxiv.org/abs/2110.09074)    | [link](https://github.com/eth-sri/fedavg_leakage)                                 |
|                      |                      | Hatamizadeh et al. (2022) [7]     | E                | FedAVG                | 224×224             | 512×8         | ResNet-18        | X-Ray Image Classification     | ★                  | ✗                                                                               |
|                      |                      | Kariyappa et al. (2023) [8] | B                | FedSGD                | 224×224             | 1024          | VGG-16           | Object Classification          | ★                  | [link](https://github.com/facebookresearch/cocktail_party_attack)                  |
|                      |                      | Usynin et al. (2023) [9] | D                | FedSGD                | 224×224             | 1             | VGG, ResNet      | Object Classification          | [2]                | ✗                                                                               |
|                      |                      | Li et al. (2023) [10]                | E                | FedSGD                | 224×224             | 256           | ResNet-50        | Object Classification          | 🆕                 | [link](https://github.com/zhaohuali/E2EGI)                                        |
|                      |                      | Ye et al. (2024) [11]                  | E                | FedSGD                | 224×224             | 8             | ResNet-34        | Object Classification          | 🆕                 | [link](https://github.com/MiLab-HITSZ/2023YeHFGradInv)                            |
|                      |                      | Li et al. (2025) [12]              | B                | FedSGD                | 224×224             | 128           | ResNet-18        | Object Classification          | [2]                | ✗                                                                               |
|                      | Augmented Optimization | Yang et al. (2023) [13] | D                | FedSGD                | 32×32               | 1             | LeNet, ResNet-18 | Object and Face Classification | [2]                | ✗                                                                               |
|                      |                      | Yue et al. (2023) [14] | D                | FedAVG                | 128×128             | 5×16          | LeNet, ResNet-18 | Object Classification          | [2]                | [link](https://github.com/KAI-YUE/rog)                                            |
|                      |                      | Sun et al. (2024) [15] | D                | FedSGD                | 64×64               | 4             | ResNet-18        | Object Classification          | ◦                  | [link](https://github.com/D1aoBoomm/GI-PIP)                                       |
|                      |                      | Liu et al. (2025) [16] | D                | FedSGD                | N/A                 | 1             | N/A              | Object Classification          | ◦                  | ✗                                                                               |
| Generative Model-based | Online Opt.         | Wang et al. (2019) [17]  | D                | FedSGD                | 64×64               | 1             | LeNet, ResNet-18 | Object and Face Classification | [2]                | ✗                                                                               |
|                      | Direct Reconstruction | Ren et al. (2022) [18] | B                | FedSGD                | 64×64               | 256           | LeNet, ResNet-18 | Object and Face Classification | ★                  | [link](https://github.com/Rand2AI/GRNN)                                           |
|                      |                      | Xue et al. (2023) [19] | D                | FedSGD                | 224×224             | 8             | ResNet-50        | Object Classification          | ◦                  | ✗                                                                               |
|                      | Latent-Space Optimization | Jeon et al. (2021) [20]                      | D                | FedSGD                | 64×64               | 4             | ResNet-18        | Object and Face Classification | ◦                  | [link](https://github.com/ml-postech/gradient-inversion-generative-image-prior)    |
|                      |                      | Li et al. (2022) [21] | D                | FedSGD                | 224×224             | 1             | ResNet-18        | Object and Face Classification | [2]                | [link](https://github.com/zhuohangli/GGL)                                         |
|                      |                      | Fang et al. (2023) [22] | D                | FedSGD                | 64×64               | 1             | ResNet-18        | Object and Face Classification | [2]                | [link](https://github.com/ffhibnese/GIFD_Gradient_Inversion_Attack)                |
|                      |                      | Xu et al. (2023) [23]  | D                | FedSGD                | 128×128             | 1             | LeNet, ResNet-18 | Object and Face Classification | [2]                | ✗                                                                               |
|                      |                      | Gu et al. (2024) [24]  | D                | FedSGD                | 256×256             | 1             | LeNet-7, ResNet-18 | Object and Face Classification | -                  | ✗                                                                               |
| Analytic-based       | Closed Form           | Zhu et al. (2021) [25]                            | B                | FedSGD                | 64×64               | 1             | 6 layer CNN      | Object Classification          | 🆕                 | [link](https://github.com/JunyiZhu-AI/R-GAP)                                      |
|                      |                      | Lu et al. (2022) [26]    | B                | FedSGD                | 224×224             | 1             | ViT              | Object Classification          | [2]                | ✗                                                                               |
|                      |                      | Dimitrov et al. (2024) [27]  | B                | FedSGD                | 256×256             | 25            | 6 layer FC-NN    | Object Classification          | -                  | ✗                                                                               |
|                      | Gradient Sparsification | Fowl et al. (2022) [28] | G                | Both                  | Input Size          | 256           | Model Agnostic²  | Object Classification          | -                  | [link](https://github.com/lhfowl/robbing_the_fed)                                 |
|                      |                      | Wen et al. (2022) [29] | F                | FedSGD                | Input Size          | 1             | Model Agnostic   | Object Classification          | -                  | ✗                                                                               |
|                      |                      | Boenisch et al. (2023) [30] | G                | Both                  | Input Size          | 100           | FC Networks³     | Object Classification          | -                  | [colab](https://colab.research.google.com/drive/17uB-plUyxNo19HVJBOKGDVWo88IiR74o?usp=sharing) |
|                      | Gradient Isolation    | Boenisch et al. (2023) [31]  | H                | FedSGD                | Input Size          | 100           | FC Networks³     | Object Classification          | -                  | ✗                                                                               |
|                      |                      | Zhao et al. (2023) [32] | G                | FedAVG                | Input Size          | 1×64          | Model Agnostic²  | Object Classification          | -                  | ✗                                                                               |
|                      |                      | Zhao et al. (2024) [33]  | G                | FedAVG                | Input Size          | 5×8           | Model Agnostic²  | Object Classification          | -                  | [link](https://github.com/Manishpandey-0/Adversarial-reconstruction-attack-on-FL-using-LOKI) |
|                      |                      | Wang et al. (2024) [34] | F                | FedSGD                | Input Size          | 100           | LeNet, VGG-16    | Object Classification          | -                  | [link](https://github.com/wfwf10/MKOR)                                            |
|                      |                      | Garov et al. (2024) [35] | G                | Both                  | Input Size          | 512           | Model Agnostic²  | Object Classification          | -                  | [link](https://github.com/insait-institute/SEER)                                  |
|                      |                      | Shi et al. (2025) [36]  | G                | Both                  | Input Size          | 1024          | Various          | Object Classification          | -                  | [link](https://github.com/unknown123489/Scale-MIA)                                |

**Symbol legend:**  
🆕 = new label recovery method  
◦ = assumes label knowledge  
★ = optimization-based label reconstruction  
- = no label restoration algorithm used  
² = adds linear layers in the shared models (potentially detectable by clients)  
³ = targets linear layers, extendable to full models by modifying preceding layers to transmit inputs  
"Batch Size" reflects the maximum used; attacks utilizing FedAVG are expressed as E×B, where E is the number of iterations/epochs.  
"Open Source" contains clickable links to available repositories.

---

**References:**  
[1] Deep Leakage from Gradients. NeurIPS. [paper](https://arxiv.org/abs/1906.08935)  

[2] iDLG: Improved Deep Leakage from Gradients. ECCV. [paper](https://arxiv.org/abs/2001.02610)  

[3] Inverting Gradients - How Easy is it to Break Privacy in Federated Learning? NeurIPS. [paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/c4ede56bbd98819ae6112b20ac6bf145-Paper.paper) 
 
[4] See Through Gradients: Image Batch Recovery via GradInversion. CVPR.  [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_See_Through_Gradients_Image_Batch_Recovery_via_GradInversion_CVPR_2021_paper.paper)

[5] GradViT: Gradient Inversion of Vision Transformers. CVPR. [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Hatamizadeh_GradViT_Gradient_Inversion_of_Vision_Transformers_CVPR_2022_paper.paper)
[6] Data Leakage in Federated Averaging. TMLR.  [paper](https://openreview.net/paper?id=e7A0B99zJf)

[7] Do Gradient Inversion Attacks Make Federated Learning Unsafe? IEEE TMI. [paper](https://ieeexplore.ieee.org/abstract/document/10025466)

[8] Cocktail Party Attack: Breaking Aggregation-based Privacy in Federated Learning Using Independent Component Analysis. ICML.  [paper](https://proceedings.mlr.press/v202/kariyappa23a/kariyappa23a.paper)

[9] Usynin, D., et al. (2023). Beyond Gradients: Exploiting Adversarial Priors in Model Inversion Attacks. ACM TOPS.  [paper](https://dl.acm.org/doi/full/10.1145/3592800)

[10] E2EGI: End-to-End Gradient Inversion in Federated Learning. IEEE JBHI.  [paper](https://ieeexplore.ieee.org/abstract/document/9878027)

[11] High-Fidelity Gradient Inversion in Distributed Learning. AAAI.  [paper](https://ojs.aaai.org/index.php/AAAI/article/view/29975)

[12] Temporal Gradient Inversion Attacks with Robust Optimization. IEEE TDSC.  [paper](https://ieeexplore.ieee.org/abstract/document/10848255)

[13] Using Highly Compressed Gradients in Federated Learning for Data Reconstruction Attacks. IEEE TIFS.  [paper](https://ieeexplore.ieee.org/abstract/document/10003066)

[14] Gradient Obfuscation Gives a False Sense of Security in Federated Learning. USENIX Security.  [paper](https://www.usenix.org/system/files/usenixsecurity23-yue.paper)

[15] GI-PIP: Do We Require Impractical Auxiliary Dataset for Gradient Inversion Attacks? ICASSP.  [paper](https://ieeexplore.ieee.org/abstract/document/10445924)

[16] Mjölnir: Breaking the Shield of Perturbation-Protected Gradients via Adaptive Diffusion. AAAI.  [paper](https://ojs.aaai.org/index.php/AAAI/article/view/34829)

[17] Beyond Inferring Class Representatives: User-Level Privacy Leakage From Federated Learning. IEEE INFOCOM.  [paper](https://ieeexplore.ieee.org/abstract/document/8737416)

[18] GRNN: Generative Regression Neural Network—A Data Leakage Attack for Federated Learning. ACM TIST.  [paper](https://dl.acm.org/doi/full/10.1145/3510032)

[19] Fast Generation-Based Gradient Leakage Attacks against Highly Compressed Gradients. IEEE INFOCOM.  [paper](https://ieeexplore.ieee.org/abstract/document/10229091)

[20] Gradient Inversion with Generative Image Prior. NeurIPS.  [paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/fa84632d742f2729dc32ce8cb5d49733-Paper.paper)

[21] Auditing Privacy Defenses in Federated Learning via Generative Gradient Leakage. CVPR. [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Auditing_Privacy_Defenses_in_Federated_Learning_via_Generative_Gradient_Leakage_CVPR_2022_paper.paper) 

[22] GIFD: A Generative Gradient Inversion Method with Feature Domain Optimization. ICCV.  [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_GIFD_A_Generative_Gradient_Inversion_Method_with_Feature_Domain_Optimization_ICCV_2023_paper.paper)

[23] CGIR: Conditional Generative Instance Reconstruction Attacks Against Federated Learning. IEEE TDSC.  [paper](https://ieeexplore.ieee.org/abstract/document/9980415)

[24] Federated Learning Vulnerabilities: Privacy Attacks with Denoising Diffusion Probabilistic Models. ACM WEB Conference.  [paper](https://dl.acm.org/doi/abs/10.1145/3589334.3645514)
[25] R-GAP: Recursive Gradient Attack on Privacy. ICLR. [paper](https://arxiv.org/abs/2104.09453)  

[26] APRIL: Finding the Achilles' Heel on Privacy for Vision Transformers. CVPR.  [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lu_APRIL_Finding_the_Achilles_Heel_on_Privacy_for_Vision_Transformers_CVPR_2022_paper.paper)

[27] SPEAR: Exact Gradient Inversion of Batches in Federated Learning. NeurIPS.  [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/c13cd7feab4beb1a27981e19e2455916-Paper-Conference.paper)
[28] Robbing the Fed: Directly Obtaining Private Data in Federated Learning with Modified Models. ICLR.  [paper](https://arxiv.org/paper/2110.13057)

[29] Fishing for User Data in Large-Batch Federated Learning via Gradient Magnification. ICML.  [paper](https://arxiv.org/paper/2202.00580)

[30] When the Curious Abandon Honesty: Federated Learning is Not Private. EuroS&P.  [paper](https://ieeexplore.ieee.org/abstract/document/10190537)

[31] Reconstructing Individual Data Points in Federated Learning Hardened with Differential Privacy and Secure Aggregation. EuroS&P.  [paper](https://ieeexplore.ieee.org/abstract/document/10190489)

[32] The Resource Problem of Using Linear Layer Leakage Attack in Federated Learning. CVPR.  [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_The_Resource_Problem_of_Using_Linear_Layer_Leakage_Attack_in_CVPR_2023_paper.paper)

[33] LOKI: Large-scale Data Reconstruction Attack Against Federated Learning Through Model Manipulation. IEEE S&P.  [paper](https://ieeexplore.ieee.org/abstract/document/10646724)
[34] Maximum Knowledge Orthogonality Reconstruction with Gradients in Federated Learning. WACV. [paper](https://openaccess.thecvf.com/content/WACV2024/papers/Wang_Maximum_Knowledge_Orthogonality_Reconstruction_With_Gradients_in_Federated_Learning_WACV_2024_paper.paper) 

[35] Hiding in Plain Sight: Disguising Data Stealing Attacks in Federated Learning. ICLR.  [paper](https://arxiv.org/paper/2306.03013)

[36] Scale-MIA: A Scalable Model Inversion Attack against Secure Federated Learning via Latent Space Reconstruction. NDSS.  [paper](https://www.ndss-symposium.org/wp-content/uploads/2025-644-paper.paper)


## Defensive Measures

| **Cat.** | **Technique**         | **Work** | **Year** | **Where** | **Threat Models** | **Intuition** | **Main Weakness** | **Open Source** |
|----------|----------------------|---------------|----------|-----------|-------------------|---------------|-------------------|-----------------|
| F        | DP-based             | [37] | N/A      | Server    | A (green) - C (green) D (orange) - E (orange) F (red) - H (red) | Server adds noise to clipped client contributions | Requires trusted (passive) server and ideal sampling conditions | ~ |
|          |                      | [38]    | N/A      | Client    | A (green) - C (green) D (orange) - H (orange) | Clients add noise to their own updates | Significantly compromises model utility; May be weakened from tailored GIAs [39, 14, 22] | ~ |
|          | Cryptography-Based   | [40]    | N/A      | Client    | A (green) - E (green) F (red) - H (red) | Server has access to aggregated client contributions only | Vulnerable to active malicious servers; Adds communication overhead | ~ |
|          |                      | [41] | N/A | Client    | A (green) - H (green) | Enables computations on encrypted data without decryption | High computational and communication overhead | ~ |
| H        | Gradient Perturbation| [42,43] | N/A      | Client    | A (green) B (orange) - C (orange) D (red) - H (red) | Transmits only the most significant gradient elements | Bypassed by modern GIAs [14, 22, 19, 16] | ~ |
|          |                      | [14]       | N/A      | Client    | A (green) B (orange) - C (orange) D (red) - H (red) | Reduces gradient precision with fewer bits | Bypassed by modern GIAs [14, 22, 19, 16] | ~ |
|          |                      | [44] | N/A  | Client    | A (green) B (orange) - C (orange) D (red) - H (red) | Limits the magnitude of gradients | Bypassed by modern GIAs [14, 22, 19, 16] | ~ |
|          |                      | [45]   | 2021     | Client    | A (green) - B (green) C (orange) D (red) - H (red) | Perturbs data representation in FC layer to modify gradient pattern | Bypassed by modern GIAs [14, 22] | [link](https://github.com/jeremy313/Soteria) |
|          |                      | [46] | 2022   | Client    | A (green) - B (green) C (orange) - H (orange) | Adds Gaussian noise to high-sensitivity components of model weights | Not tested against recent generative model-based GIAs | [link](https://github.com/wangjunxiao/GradDefense) |
|          |                      | [47]   | 2024     | Client    | A (green) - B (green) C (orange) - H (orange) | Adaptive noise injection with sensitivity-informed perturbation strategy | Not tested against recent generative model-based GIAs | ✗ |
|          |                      | [48]    | 2025     | Client    | A (green) - D (green) E (orange) - H (orange) | Perturb gradients in a subspace orthogonal to the original one | Not evaluated against attack with stronger threat model | [link](https://github.com/KaiyuanZh/censor) |
| H        | Learning Algorithm Modification | DigestNN [49] | 2021 | Client | A (green) - B (green) C (orange) - H (orange) | Transforms data into dissimilar representations | Not tested against generative model-based GIAs | ✗ |
|          |                      | [50]       | 2022     | Client    | A (green) - B (green) C (orange) - H (orange) | Slices and encrypts gradients between clients | Not tested against generative model-based GIAs | [link](https://github.com/najeebjebreel/FFL) |
|          |                      | [51]       | 2022     | Client    | A (green) - D (green) E (orange) - H (orange) | Dynamically modifies learning rate for each client to make gradient estimation difficult | Uncertain impact on optimization dynamics | ✗ |
|          |                      | [52]       | 2023     | Client    | A (green) - B (green) C (orange) - H (orange) | Uses augmentation to balance privacy and utility | Vulnerable during early training phases [53] | [link](https://github.com/gaow0007/ATSPrivacy) |
|          |                      | PEFL [54]      | 2023     | Client    | A (green) - B (green) C (orange) - H (orange) | Decomposes weight matrices into cascading sub-matrices creating nonlinear mapping between gradients and raw data | Not tested against generative model-based GIAs | ✗ |
|          |                      | GIAnDe [55]    | 2024     | Client    | A (green) - B (green) C (orange) - H (orange) | Plug-and-play defense using vicinal distribution augmentation of training data | Not tested against generative model-based GIAs | [link](https://github.com/MiLab-HITSZ/2023YeGIAnDe) |
|          |                      | DCS-2 [56]     | 2024     | Client    | A (green) - B (green) C (orange) D (green) <br> E (orange) F (green) G (orange) - H (orange) | Use visually different synthesized concealed samples to compute model updates | Introduce computational overhead to synthesize concealed images | [link](https://github.com/JingWu321/DCS-2) |
| H        | Model Modification   | SPN [57]       | 2020     | Client    | A (green) - B (green) C (orange) - H (orange) | Parallel branch with weights hidden from server | May be vulnerable to branch simulation scenarios or recent GIAs | ✗ |
|          |                      | PRECODE [58]   | 2022     | Client    | A (green) - B (green) C (orange) - H (orange) | Variational block adding randomness | Proven ineffective against advanced GIAs [14] | [link](https://github.com/dAI-SY-Group/PRECODE) |
|          |                      | FedKL [59]     | 2023     | Client    | A (green) B (orange) - H (orange) | Extends model with branch hidden from server | May be vulnerable to branch simulation scenarios or recent GIAs | [link](https://github.com/Rand2AI/FedKL) |

---

**References:**  
[37] Learning Differentially Private Recurrent Language Models. [paper](https://arxiv.org/abs/1710.06963)
[38] FedCDP: Client-level Differential Privacy for Federated Learning. IEEE ICDCS [paper](https://ieeexplore.ieee.org/abstract/document/9546481)
[39] Does Differential Privacy Really Protect Federated Learning From Gradient Leakage Attacks? [paper](https://ieeexplore.ieee.org/abstract/document/10568968)
[40] Sok: Secure aggregation based on cryptographic schemes for federated learning. [paper](https://petsymposium.org/popets/2023/popets-2023-0009.php#:~:text=popets%2D2023%2D0009-,Download%20paper,-Abstract%3A%20Secure)
[41] Practical Secure Aggregation for Privacy-Preserving Machine Learning. CCS.  [paper](https://dl.acm.org/doi/abs/10.1145/3133956.3133982)
[42] Revisiting Gradient Pruning: A Dual Realization for Defending against Gradient Attacks. AAAI [paper](https://ojs.aaai.org/index.php/AAAI/article/view/28460)
[43] Preserving data privacy in federated learning through large gradient pruning. CoSe [paper](https://www.sciencedirect.com/science/article/pii/S016740482200431X)
[44] Auditing Privacy Defenses in Federated Learning via Generative Gradient Leakage [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Auditing_Privacy_Defenses_in_Federated_Learning_via_Generative_Gradient_Leakage_CVPR_2022_paper.paper)
[45] Soteria: Provable Defense Against Privacy Leakage in Federated Learning. [GitHub](https://github.com/jeremy313/Soteria)  [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_Soteria_Provable_Defense_Against_Privacy_Leakage_in_Federated_Learning_From_CVPR_2021_paper.paper)
[46] Protect Privacy from Gradient Leakage Attack in Federated Learning [GitHub](https://github.com/wangjunxiao/GradDefense)  [paper](https://ieeexplore.ieee.org/abstract/document/9796841)
[47] More Than Enough is Too Much: Adaptive Defenses Against Gradient Leakage in Production Federated Learning [paper](https://ieeexplore.ieee.org/document/10477938)
[48] CENSOR: Defense Against Gradient Inversion via Orthogonal Subspace Bayesian Sampling. NDSS 2025 [GitHub](https://github.com/KaiyuanZh/censor) [paper](https://kaiyuanzhang.com/publications/NDSS25_Censor.paper)
[49] Digestive neural networks: A novel defense strategy against inference attacks in federated learning [paper](https://www.sciencedirect.com/science/article/pii/S0167404821002029)
[50] Enhanced Security and Privacy via Fragmented Federated Learning [GitHub](https://github.com/najeebjebreel/FFL)  [paper](https://ieeexplore.ieee.org/abstract/document/9925189)
[51] Enhancing Privacy Preservation in Federated Learning via Learning Rate Perturbation. [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wan_Enhancing_Privacy_Preservation_in_Federated_Learning_via_Learning_Rate_Perturbation_ICCV_2023_paper.paper)
[52] Automatic Transformation Search Against Deep Leakage from Gradients [GitHub](https://github.com/gaow0007/ATSPrivacy)  [paper](https://ieeexplore.ieee.org/document/10086616)
[53] Bayesian Framework for Gradient Leakage [paper](https://arxiv.org/paper/2111.04706) 
[54] Privacy-Encoded Federated Learning Against Gradient-Based Data Reconstruction Attacks [paper](https://ieeexplore.ieee.org/abstract/document/10231369)
[55] Gradient Inversion Attacks: Impact Factors Analyses and Privacy Enhancement [GitHub](https://github.com/MiLab-HITSZ/2023YeGIAnDe)  [paper](https://ieeexplore.ieee.org/document/10604429)
[56] Concealing Sensitive Samples against Gradient Leakage in Federated Learning [GitHub](https://github.com/JingWu321/DCS-2)  [paper](https://ojs.aaai.org/index.php/AAAI/article/view/30171)
[57] Rethinking Privacy Preserving Deep Learning: How to Evaluate and Thwart Privacy Attacks [paper](https://link.springer.com/chapter/10.1007/978-3-030-63076-8_3)
[58] PRECODE - A Generic Model Extension To Prevent Deep Gradient Leakage [GitHub](https://github.com/dAI-SY-Group/PRECODE) [paper](https://openaccess.thecvf.com/content/WACV2022/html/Scheliga_PRECODE_-_A_Generic_Model_Extension_To_Prevent_Deep_Gradient_WACV_2022_paper.html#:~:text=%5B-,paper,-%5D%20%5Bsupp%5D) 
[59] Gradient Leakage Defense with Key-Lock Module for Federated Learning [GitHub](https://github.com/Rand2AI/FedKL)  [paper](https://arxiv.org/abs/2305.04095)

---
# Citation

If you find this repository useful for your research, please consider citing our work:

```bibtex
@inproceedings{carletti2025sok,
  title     = {SoK: Gradient Inversion Attacks in Federated Learning},
  author    = {Vincenzo Carletti, Pasquale Foggia, Carlo Mazzocca, Giuseppe Parrella and Mario Vento},
  booktitle = {34th USENIX Security Symposium (USENIX Security 25)},
  year      = {2025},
}
```
---

