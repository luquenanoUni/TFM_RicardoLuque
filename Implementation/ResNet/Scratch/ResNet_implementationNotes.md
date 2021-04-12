Implementation
=====================
Skip connections to overcome optimization problems in weights updates when backward propagating.

# LFW Dataset
Centered and aligned using an implementation of the Viola-Jones face detector


# Inception V3 Tryout
- Did not work due to input size of images. Inception needs an upsampled version of the input for which resizing a 70x47 input image will definitely result in low performance given the noise or sparsity induced by interpolation.

# ResNet 50 transfer learning
- Training accuracy does improve, but validation doesn't increase. It gets stuck
  - It may be due to split of validation and training. There's a random seed=42 that has been introduced to 
- Change the minimum samples of images each class has
- Increase accuracy by adapting learning rate throughout the epochs
- Change the split of validation and training
- Intenta hacerlo de manera más científica, documentando con razonamiento las ideas de por qué no funciona. Más trial and error documentado

# Configurations
- **Choosing ADAM instead of SGD or RMSprop**: When certain features are more recurrent than others, then the same learning rate for all parameters is not a good idea. A larger update for rarely occurring features is necessary, whereas not so necessary for frequently occurring ones. 
  - Adam uses both: exponentially decaying average of the past squared gradients and the exponentially decaying average of past gradients, to update weights
  - The RMSprop optimizer is similar to the gradient descent algorithm with momentum. The RMSprop optimizer restricts the oscillations in the vertical direction. Therefore, we can increase our learning rate and our algorithm could take larger steps in the horizontal direction converging faster. Both of them SGD and RMSprop are likely to get  stuck in a local minima. [Link](https://towardsdatascience.com/a-look-at-gradient-descent-and-rmsprop-optimizers-f77d483ef08b)
- **Choosing the appropriate LR** We try to dwindle around the minimum and if we choose a very small learning rate the convergence gets really slow. In the neural networks domain, one of the issues we face with the highly non convex functions is that one gets trapped in the numerous local minimas.


mini tutorial que quizás ayude a dar ideas: https://machinelearningmastery.com/introduction-to-deep-learning-for-face-recognition/#:~:text=Face%20recognition%20is%20often%20described,extraction%2C%20and%20finally%20face%20recognition.

https://machinelearningmastery.com/one-shot-learning-with-siamese-networks-contrastive-and-triplet-loss-for-face-recognition/

Journal
Face tracking for a given set of time