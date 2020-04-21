# dlgfa

We compared DLGFA with VAE[1],CVAE[2],oi-VAE[3],and GFA[4]. To further check and validate how well the disentanglement is achieved, we propose to visualize the W matrix and quantitatively compare the MSE (mean squared error) on the test data with alternative methods. 

1. Synthetic image data: We generated 2000 8X8 one bar images, and randomly select 80% of the image for training (n=1600), 10% for validation (n=200) and 10% for testing (n=200). The row positions of the bar was taken as different time point labels, for example, if one bar appeared in the first row of the 8X8 image we label this image belong to time point t=1. Latent dimensions Z=[4,8,16].
Results: Mean squared error on test set. The numbers in the parentless represent standard error of per-sample MSE. Z=4: VAE[0.0301(0.0022], CVAE[0.0026(0.0004)], oi-VAE(0.03110.0129), DLGFA(0.00320.0011) 
Z=8: VAE(0.02900.0022), CVAE(0.00250.0004), oi-VAE(0.00510.0008), DLGFA(0.00280.0002) 
Z=16: VAE(0.03100.0036), CVAE(0.00270.0005), oi-VAE(0.00520.0009), DLGFA(0.00310.0001)
From the MSE results, CAVE and DLGFA performs similarly and significant different (better) than VAE and oi-VAE.

2. Motion capture data: We used the trial 1 to 10 as the training set, and trial 11 as the test set.  For CVAE, we use total time point T=32 to label the frames, total 32  118 frames.
Z=4: VAE(0.08970.0160), CVAE(0.05600.0025), oi-VAE(0.06090.0059), DLGFA(0.04620.0062) 
Z=8: VAE(0.08960.0163), CVAE(0.04250.0040), oi-VAE(0.05690.0058), DLGFA(0.03260.0056) 
Z=16: VAE(0.08990.0168), CVAE(0.04710.0050), oi-VAE(0.0520.0055), DLGFA(0.03840.0010)

From the MSE results, under complex motion capture data, DLGFA performs better than all the competitors. 

3. Metabolomics data:  we have limited sample size, n= 8 (this is common issue for longitudinal clinical study), t=12, we randomly select 6 subjects as training set and the rest 2 subjects as test set for DLGFA, while VAE and oi-VAE take the inputs as independent.  We have five groups, each group has 196 features.
Z=10: VAE(0.09650.0625), CVAE(0.07420.0409), oi-VAE(0.04640.0250), DLGFA(0.04120.0030) 
Z=20: VAE(0.08080.0168), CVAE(0.05220.0050), oi-VAE(0.06180.0309), DLGFA(0.03840.0041)
Z=30: VAE(0.07770.0473), CVAE(0.04670.0257), oi-VAE(0.04790.0202), DLGFA(0.02590.0067) 
DLGFA performs better than all the competitors under the limited sample size data.








[1] Kingma, D. P. and Welling, M. Auto-encoding variational bayes. arXiv:1312.6114, 2013. 
[2] Sohn et al., Learning Structured Output Representation Using Deep Conditional Generative Models. NIPS 2015.
[3] Ainsworth, S. K., Foti, N. J., Lee, A. K. C., and Fox, E. B. oi-vae: Output interpretable vaes for nonlinear group
factor analysis. In Proceedings of the 35th International Conference on Machine Learning (ICML 18), 2018.
[4] Leppaaho, E., Ammad–ud–din, M., and Kaski, S. Gfa: Exploratory analysis of multiple data sources with group
factor analysis. Journal of Machine Learning Research, 18:1–5, 2017.
[5] Casale et al. Gaussian Process Prior Variational Autoencoders, NeurIPS 2018.
