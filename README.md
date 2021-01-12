# NNSE

This is the implementation code for NNSE.

Reference to be updated: 

"Neural Networks Embedded Self-Expression with Adaptive Graph for Unsupervised Feature Selection", Aihong Yuan, Mengbo You*

Unsupervised feature selection (UFS), which unsupervisely removes the redundant information and selects the most representative feature subset for subsequent data analysis, is a fundamental problem in machine learning and has been studied for many years. Most UFS methods map features into a pseudo label space by multiplying a projection matrix constrained with sparsity to learn the mapping from the features to the labels. However, the mapping relationship is usually not linear, and linear regression may result in a suboptimal selection. To address this issue, we propose a novel UFS method, called neural networks embedded self-expression (NNSE). NNSE replaces the linear regression of traditional spectral analysis methods with neural networks to learn the pseudo label space. Besides, we embeds neural networks into the self-expression model to improve the representative ability by preserving the local structure with an adaptive graph regularization module. Then we propose an efficient alternative iterative algorithm to solve the proposed model. Experimental results on 8 public datasets show that NNSE outperforms 8 comparative UFS methods in terms of clustering accuracy. Moreover, we also present experimental results on convergence of the objective function, ablation study, visualization of the data point distribution, and the feature selection priority.
