# CNN Optimization
A repository for code on optimizing CNN models for the identification of images of asteroids resonant angles.

The asteroidal main belt is crossed by a web of mean-motion and secular resonances, that occur when there is a commensurability between fundamental frequencies of the asteroids and planets.  Traditionally, these objects were identified by visual inspection of the time evolution of their resonant argument, which is a combination of orbital elements of the asteroid and the perturbing planet(s). Since the population of asteroids affected by these resonances is, in some cases, of the order of several thousand, this has become a taxing task for a human observer.  
Recent works used Convolutional Neural Networks (CNN) models to perform such task automatically.  In this work, we compare the outcome of such models with those of some of the most advanced and publicly available CNN architectures, like the VGG, Inception and ResNet. The performance of such models is first tested and optimized for overfitting issues, using validation sets and a series of regularization techniques like data augmentation, dropout, and batch normalization. The three best-performing models were then used to predict the labels of larger testing databases containing thousands of images.
In this repository we provide the Python codes for testing the VGG, Inception and ResNet models over databases of thousands of images of resonant arguments of asteroids interacting with the M1:2 exterior mean-motion resonance with Mars (CNN_model_M12.py), and for nu6 secular resonance with Saturn (CNN_model_nu6.py).
The best-performing model can be loaded using Load_best_model_M12.py and Load_best_model_nu6.py, and can be used to predict the labels of thousands of images in seconds.
Repositories of the images and labels used in this work are available at these links:

https://drive.google.com/file/d/1RsDoMh8iMwZhD-fnkYSs9hiWmg96SZf0/view?usp=sharing

https://drive.google.com/drive/folders/1oxdXquibdYYP965AgzoMjFop-ZZXcHni

Folders containing the model weights and the image repositories for the best CNN models for the M1:2 and nu6 resonances
can be accessed at this link:

https://drive.google.com/drive/folders/1S0yGI5dbwCu1VV6nimRWnvPQ3JFKBm5O?usp=share_link

More information on this work is available on Carruba et al. 2022, CMDA, under review, aslo available on ArXiv:

https://ui.adsabs.harvard.edu/link_gateway/2022arXiv220714181C/arxiv:2207.14181
