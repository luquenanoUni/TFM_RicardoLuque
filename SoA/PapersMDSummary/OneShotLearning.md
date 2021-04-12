One shot learning (Blog)
=================
[Link](https://machinelearningmastery.com/one-shot-learning-with-siamese-networks-contrastive-and-triplet-loss-for-face-recognition/)
Based on **Siamese networks** using **contrastive loss**, which computes the performance of the model between pairs of inputs instead of across all input examples in the training set. Therefore, pairs of examples are provided to the network and the penalization occurs based on whether the classes of the samples are the same or different. 
- If classes are the **same** the loss function encourages models to output feature vectors that are more similar between them
- If classes are **different** the loss fucntion encourages models to ouput feature vectors that are less similar.

## Siamese networks: 
Siamese networks are an approach to addressing one-shot learning in which a learned feature vector for the known and candidate example are compared.

## Contrastive loss
It requires face image pairs and then pulls together positive pairs and pushes apart negative pairs. 
- *Problem:* margin parameters are often difficult to choose. Contrastive loss requires that a margin is selected in order to determine the limit to which examples from different pairs are penalized

Siamese networks may be avoided by providing pairs of examples sequentially and saving the predicted feature vectors before calculating the loss and updating the model. *eg: DeepL2 & DeepL3*

## Triplet loss:
Is an extention of Contrastive loss, but by using *3 samples* instead of 2.

Involves: 
1. Anchor example 
2. Matching example: Same class 
3. Non-matching example: Different class

Penalizes the model such that the distance between the matching examples is reduced and the distance between the non-matching examples is increased.

The result is a *'face embedding* feature vector that has a meaningful Euclidean relationship, i.e. similar faces produce embeddings that have **small distances** *(e.g. can be clustered)* and different examples of the same face produce embeddings that are very small and allow verification and discrimination from other identities.

### Choosing my set of triplets:
Hard triplets are sought that encourage changes to the model and the predicted face embeddings.


Siamese Neural Networks For One SHot Learning (Paper) 2015
==========================================================
# IMPORTANT: THIS ONE IS FOR CHARACTER REGOGNITION, however it may be used for image classification 
::PROS::
- Powerful when little data is available
- Able to generalize to unfamiliar categories without extensive retraining
- Easily trainable

::CONTRIBUTION::
- Overcomes developing **domain-specific** features or inference procedures which possess highly discriminative properties for the target task
- Limits assumptions on the structure of the inputs
# Approach
1. Capable of learning generic image features useful for making predictions from unknown class distributions
2. Uses standard optimization techniques on pairs sampled from the source data
3. Provide an approach that does not rely upon domain-specific knowledge

**Verification task** *We need to develop a NN that can discriminate between class-identity of image pairs* According to the probability of belonging to the same class or not

Siamese networks enforce similar mapping of feature vectors on similar images.

Using cross entropy for training the network.

Using stochastic gradient descent with momemtum

Further reading
===============
Zero shot learning
