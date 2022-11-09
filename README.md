# CNN Optimization
A repository for code on optimizing CNN models for the identification of images of asteroids resonant angles.

The asteroidal main belt is crossed by a web of mean-motion and secular resonances, that occur when there is a commensurability between fundamental frequencies of the asteroids and planets.  Traditionally, these objects were identified by visual inspection of the time evolution of their resonant argument, which is a combination of orbital elements of the asteroid and the perturbing planet(s). Since the population of asteroids affected by these resonances is, in some cases, of the order of several thousand, this has become a taxing task for a human observer.  
Recent works used Convolutional Neural Networks (CNN) models to perform such task automatically.  In this work, we compare the outcome of such models with those of some of the most advanced and publicly available CNN architectures, like the VGG, Inception and ResNet. The performance of such models is first tested and optimized for overfitting issues, using validation sets and a series of regularization techniques like data augmentation, dropout, and batch normalization.
The three best-performing models were then used to predict the labels of larger testing databases containing thousands of images.
In this repository we provide the codes for testing the VGG, Inception and ResNet models over databases of thousands of images of resonant arguments of asteroids interacting with the nu6 secular resonance (CNN_model_nu6.py).
The best-performing model can be loaded using Load_best_model_nu6.py, and can be used to predict the labels of thousands of images in seconds.
Repositories of the images and labels used in this work are available upon reasonable request.
More information on this work is available on Carruba et al. 2022, CMDA, under review.
