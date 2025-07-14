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
[1] Zhu, L., Liu, Z., & Han, S. (2019). Deep Leakage from Gradients. NeurIPS. [pdf](https://arxiv.org/abs/1906.08935)  
[2] Zhao, B., Mopuri, K. R., & Bilen, H. (2020). iDLG: Improved Deep Leakage from Gradients. ECCV. [pdf](https://arxiv.org/abs/2001.02610)  
[3] Geiping, J., Bauermeister, H., Dröge, H., & Moeller, M. (2020). Inverting Gradients - How Easy is it to Break Privacy in Federated Learning? NeurIPS. [pdf](https://arxiv.org/abs/2004.10397)  
[4] Yin, H., Chen, Y., Wang, S., et al. (2021). See Through Gradients: Image Batch Recovery via GradInversion. CVPR.  
[5] Hatamizadeh, A., Yang, D., et al. (2022). GradViT: Gradient Inversion of Vision Transformers. CVPR.  
[6] Dimitrov, D., et al. (2022). Data Leakage in Federated Averaging. TMLR.  
[7] Hatamizadeh, A., Yang, D., et al. (2022). Do Gradient Inversion Attacks Make Federated Learning Unsafe? IEEE TMI.  
[8] Kariyappa, S., et al. (2023). Cocktail Party Attack: Breaking Aggregation-based Privacy in Federated Learning Using Independent Component Analysis. ICML.  
[9] Usynin, D., et al. (2023). Beyond Gradients: Exploiting Adversarial Priors in Model Inversion Attacks. ACM TOPS.  
[10] Li, Z., et al. (2023). E2EGI: End-to-End Gradient Inversion in Federated Learning. IEEE JBHI.  
[11] Ye, H., Luo, X., Zhou, Y., Tang, J. (2024). High-Fidelity Gradient Inversion in Distributed Learning. AAAI.  
[12] Li, Z., et al. (2025). Temporal Gradient Inversion Attacks with Robust Optimization. IEEE TDSC.  
[13] Yang, Y., et al. (2023). Using Highly Compressed Gradients in Federated Learning for Data Reconstruction Attacks. IEEE TIFS.  
[14] Yue, K., et al. (2023). Gradient Obfuscation Gives a False Sense of Security in Federated Learning. USENIX Security.  
[15] Sun, D., et al. (2024). GI-PIP: Do We Require Impractical Auxiliary Dataset for Gradient Inversion Attacks? ICASSP.  
[16] Liu, Y., et al. (2025). Mjölnir: Breaking the Shield of Perturbation-Protected Gradients via Adaptive Diffusion. AAAI.  
[17] Wang, S., et al. (2019). Beyond Inferring Class Representatives: User-Level Privacy Leakage From Federated Learning. IEEE INFOCOM.  
[18] Ren, Y., et al. (2022). GRNN: Generative Regression Neural Network—A Data Leakage Attack for Federated Learning. ACM TIST.  
[19] Xue, Y., et al. (2023). Fast Generation-Based Gradient Leakage Attacks against Highly Compressed Gradients. IEEE INFOCOM.  
[20] Jeon, Y., et al. (2021). Gradient Inversion with Generative Image Prior. NeurIPS.  
[21] Li, Z., et al. (2022). Auditing Privacy Defenses in Federated Learning via Generative Gradient Leakage. CVPR.  
[22] Fang, Y., et al. (2023). GIFD: A Generative Gradient Inversion Method with Feature Domain Optimization. ICCV.  
[23] Xu, J., et al. (2023). CGIR: Conditional Generative Instance Reconstruction Attacks Against Federated Learning. IEEE TDSC.  
[24] Gu, X., et al. (2024). Federated Learning Vulnerabilities: Privacy Attacks with Denoising Diffusion Probabilistic Models. WWW.  
[25] Zhu, J., et al. (2021). R-GAP: Recursive Gradient Attack on Privacy. ICLR. [pdf](https://arxiv.org/abs/2104.09453)  
[26] Lu, Y., et al. (2022). APRIL: Finding the Achilles' Heel on Privacy for Vision Transformers. CVPR.  
[27] Dimitrov, D., et al. (2024). SPEAR: Exact Gradient Inversion of Batches in Federated Learning. NeurIPS.  
[28] Fowl, L., et al. (2022). Robbing the Fed: Directly Obtaining Private Data in Federated Learning with Modified Models. ICLR.  
[29] Wen, Y., et al. (2022). Fishing for User Data in Large-Batch Federated Learning via Gradient Magnification. ICML.  
[30] Boenisch, T., et al. (2023). When the Curious Abandon Honesty: Federated Learning is Not Private. EuroS&P.  
[31] Boenisch, T., et al. (2023). Reconstructing Individual Data Points in Federated Learning Hardened with Differential Privacy and Secure Aggregation. EuroS&P.  
[32] Zhao, M., et al. (2023). The Resource Problem of Using Linear Layer Leakage Attack in Federated Learning. CVPR.  
[33] Zhao, M., et al. (2024). LOKI: Large-scale Data Reconstruction Attack Against Federated Learning Through Model Manipulation. IEEE S&P.  
[34] Wang, F., et al. (2024). Maximum Knowledge Orthogonality Reconstruction with Gradients in Federated Learning. WACV.  
[35] Garov, G., et al. (2024). Hiding in Plain Sight: Disguising Data Stealing Attacks in Federated Learning. ICLR.  
[36] Shi, Y., et al. (2025). Scale-MIA: A Scalable Model Inversion Attack against Secure Federated Learning via Latent Space Reconstruction. NDSS.  


## Defensive Measures

| **Cat.** | **Technique**         | **Work** [Ref] | **Year** | **Where** | **Threat Models** | **Intuition** | **Main Weakness** | **Open Source** |
|----------|----------------------|---------------|----------|-----------|-------------------|---------------|-------------------|-----------------|
| F        | DP-based             | DP-FedAvg [37] | N/A      | Server    | A (green) - C (green) D (orange) - E (orange) F (red) - H (red) | Server adds noise to clipped client contributions | Requires trusted (passive) server and ideal sampling conditions | ~ |
|          |                      | FedCDP [38]    | N/A      | Client    | A (green) - C (green) D (orange) - H (orange) | Clients add noise to their own updates | Significantly compromises model utility; May be weakened from tailored GIAs [39, 14, 22] | ~ |
|          | Cryptography-Based   | SoK-SA [40]    | N/A      | Client    | A (green) - E (green) F (red) - H (red) | Server has access to aggregated client contributions only | Vulnerable to active malicious servers; Adds communication overhead | ~ |
|          |                      | Bonawitz et al. [41] | N/A | Client    | A (green) - H (green) | Enables computations on encrypted data without decryption | High computational and communication overhead | ~ |
| H        | Gradient Perturbation| Pruning [42,43] | N/A      | Client    | A (green) B (orange) - C (orange) D (red) - H (red) | Transmits only the most significant gradient elements | Bypassed by modern GIAs [14, 22, 19, 16] | ~ |
|          |                      | ROG [14]       | N/A      | Client    | A (green) B (orange) - C (orange) D (red) - H (red) | Reduces gradient precision with fewer bits | Bypassed by modern GIAs [14, 22, 19, 16] | ~ |
|          |                      | Audit Defense [44] | N/A  | Client    | A (green) B (orange) - C (orange) D (red) - H (red) | Limits the magnitude of gradients | Bypassed by modern GIAs [14, 22, 19, 16] | ~ |
|          |                      | Soteria [45]   | 2021     | Client    | A (green) - B (green) C (orange) D (red) - H (red) | Perturbs data representation in FC layer to modify gradient pattern | Bypassed by modern GIAs [14, 22] | [link](https://github.com/jeremy313/Soteria) |
|          |                      | GradDefense [46] | 2022   | Client    | A (green) - B (green) C (orange) - H (orange) | Adds Gaussian noise to high-sensitivity components of model weights | Not tested against recent generative model-based GIAs | [link](https://github.com/wangjunxiao/GradDefense) |
|          |                      | Outpost [47]   | 2024     | Client    | A (green) - B (green) C (orange) - H (orange) | Adaptive noise injection with sensitivity-informed perturbation strategy | Not tested against recent generative model-based GIAs | ✗ |
|          |                      | Censor [48]    | 2025     | Client    | A (green) - D (green) E (orange) - H (orange) | Perturb gradients in a subspace orthogonal to the original one | Not evaluated against attack with stronger threat model | [link](https://github.com/KaiyuanZh/censor) |
| H        | Learning Algorithm Modification | DigestNN [49] | 2021 | Client | A (green) - B (green) C (orange) - H (orange) | Transforms data into dissimilar representations | Not tested against generative model-based GIAs | ✗ |
|          |                      | FFL [50]       | 2022     | Client    | A (green) - B (green) C (orange) - H (orange) | Slices and encrypts gradients between clients | Not tested against generative model-based GIAs | [link](https://github.com/najeebjebreel/FFL) |
|          |                      | LRP [51]       | 2022     | Client    | A (green) - D (green) E (orange) - H (orange) | Dynamically modifies learning rate for each client to make gradient estimation difficult | Uncertain impact on optimization dynamics | ✗ |
|          |                      | ATS [52]       | 2023     | Client    | A (green) - B (green) C (orange) - H (orange) | Uses augmentation to balance privacy and utility | Vulnerable during early training phases [53] | [link](https://github.com/gaow0007/ATSPrivacy) |
|          |                      | PEFL [54]      | 2023     | Client    | A (green) - B (green) C (orange) - H (orange) | Decomposes weight matrices into cascading sub-matrices creating nonlinear mapping between gradients and raw data | Not tested against generative model-based GIAs | ✗ |
|          |                      | GIAnDe [55]    | 2024     | Client    | A (green) - B (green) C (orange) - H (orange) | Plug-and-play defense using vicinal distribution augmentation of training data | Not tested against generative model-based GIAs | [link](https://github.com/MiLab-HITSZ/2023YeGIAnDe) |
|          |                      | DCS-2 [56]     | 2024     | Client    | A (green) - B (green) C (orange) D (green) <br> E (orange) F (green) G (orange) - H (orange) | Use visually different synthesized concealed samples to compute model updates | Introduce computational overhead to synthesize concealed images | [link](https://github.com/JingWu321/DCS-2) |
| H        | Model Modification   | SPN [57]       | 2020     | Client    | A (green) - B (green) C (orange) - H (orange) | Parallel branch with weights hidden from server | May be vulnerable to branch simulation scenarios or recent GIAs | ✗ |
|          |                      | PRECODE [58]   | 2022     | Client    | A (green) - B (green) C (orange) - H (orange) | Variational block adding randomness | Proven ineffective against advanced GIAs [14] | [link](https://github.com/dAI-SY-Group/PRECODE) |
|          |                      | FedKL [59]     | 2023     | Client    | A (green) B (orange) - H (orange) | Extends model with branch hidden from server | May be vulnerable to branch simulation scenarios or recent GIAs | [link](https://github.com/Rand2AI/FedKL) |

---

**References:**  
[37] DP-FedAvg: Differential Privacy for Federated Averaging.  
[38] FedCDP: Client-level Differential Privacy for Federated Learning.  
[39] GLA-DP: Gradient Leakage Attacks against Differentially Private Federated Learning.  
[40] SoK-SA: Secure Aggregation for Federated Learning.  
[41] Bonawitz, K., et al. (2017). Practical Secure Aggregation for Privacy-Preserving Machine Learning. CCS.  
[42] Pruning: Gradient Pruning for Privacy in Federated Learning.  
[43] Pruning2: Additional work on gradient pruning.  
[44] Audit Defense: Gradient Magnitude Limiting for Privacy.  
[45] Soteria: Zhu, L., et al. (2021). Soteria: Provable Defense Against Privacy Leakage in Federated Learning. [GitHub](https://github.com/jeremy313/Soteria)  
[46] GradDefense: Wang, J., et al. (2022). GradDefense: Defense Against Gradient Leakage in Federated Learning. [GitHub](https://github.com/wangjunxiao/GradDefense)  
[47] Outpost: Adaptive Noise Injection for Federated Learning.  
[48] Censor: Zhang, K., et al. (2025). Censor: Orthogonal Gradient Perturbation for Privacy. [GitHub](https://github.com/KaiyuanZh/censor)  
[49] DigestNN: Digest Neural Networks for Privacy.  
[50] FFL: Federated Feature Learning. [GitHub](https://github.com/najeebjebreel/FFL)  
[51] LRP: Learning Rate Perturbation for Privacy.  
[52] ATS: Gao, W., et al. (2023). ATSPrivacy: Augmentation-based Training for Privacy. [GitHub](https://github.com/gaow0007/ATSPrivacy)  
[53] Balunovic, M., et al. (2021). Bayesian Privacy Analysis of Federated Learning.  
[54] PEFL: Privacy-Enhanced Federated Learning.  
[55] GIAnDe: Ye, H., et al. (2024). GIAnDe: Vicinal Distribution Augmentation for Privacy. [GitHub](https://github.com/MiLab-HITSZ/2023YeGIAnDe)  
[56] DCS-2: Wu, J., et al. (2024). DCS-2: Concealed Sample Synthesis for Privacy. [GitHub](https://github.com/JingWu321/DCS-2)  
[57] SPN: Split Neural Networks for Privacy.  
[58] PRECODE: PRECODE: Variational Block for Privacy. [GitHub](https://github.com/dAI-SY-Group/PRECODE)  
[59] FedKL: Ren, Y., et al. (2023). FedKL: Federated Learning with Hidden Branches. [GitHub](https://github.com/Rand2AI/FedKL)  
[14] Yue, K., et al. (2023). Gradient Obfuscation Gives a False Sense of Security in Federated Learning. USENIX Security.  
[16] Liu, Y., et al. (2025). Mjölnir: Breaking the Shield of Perturbation-Protected Gradients via Adaptive Diffusion. AAAI.  
[19] Xue, Y., et al. (2023). Fast Generation-Based Gradient Leakage Attacks against Highly Compressed Gradients. IEEE INFOCOM.  
[22] Fang, Y., et al. (2023). GIFD: A Generative Gradient Inversion Method with Feature Domain Optimization. ICCV.  


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

