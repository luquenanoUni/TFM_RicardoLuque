# Face recognition - A One-shot learning perspective

Introduction
============

Traditional Deep Learning based methods rely on having a huge number of annotated training samples per class.

::Contribution::
- Combining the best of deep learned features with a traditional One-shot learning framework
- Siamese NN-based approach to perform One-shot learning and classification
- Hybrid Siamese NN with Res-Net encoded features for One-shot face recognition
- Deep Convolutional Siamese network and transfer learning strategy to produce robust face recognition system which leverages the deep learned features attributes.

::Datasets::
2 publicly available datasets
DNN-based features from DLIB-ml machine learning toolkit are used for feature representation
Indian Movie Faces Database (IMFDB)

::Performance::
90% on 5-way One-shot tasks
84% on 50-way One-shot problems

### Back in the day
Used to rely on hand-crafted features like SIFT, SURF; LBP, HoG features, which **failed to counter challenges of unconstrained face recognition.** However, they did address specifics, such as changes in lighting, pose and expression.

### Nowadays
Deep learning is based on learning multiple representations and abstractions by using a cascade of processing units for feature extraction and transformation. 

::Classical DL Drawbacks::
- Demands for a huge amount of annotated data to train the system and the requirement of retraining when a new class is added.

::One-shot benefits::
Performs a classification seeing only a handful of training samples. 

 
Related work
============
Face recognition methods be divided into 2 groups
1. _Handcrafted features:_ Focused mainly on high-dimensional artifical feature extraction and feature vector dimensionality reduction using PCA or Linear Discriminant Analysis.
2. _Deep-learned features-based (DLFP)_: Directly from the image. 
::CONS DLFB::
- struggle to deal with many real-world applications
- bad performance when small amounts of data or class imbalance is present
- Not robust at scale

::Dealing with imbalance:: 
- Guo et al. Underrepresented classes promotion loss term to align norms of weights of vectors from underrepresented and normal classes. 
- Want et al proposes a CNN-Based framework which deals with deficient training data by using a balancing regulariser and shifting the center regeneration to regulate weight vector norms.
- Ding et al. proposed focusing on building generative models to build extra examples. A generative model to synthesize data for one-shot classes by adapting the data variances and augmenting features from other classes.
- Jhadav et al proposed using specific attributes of human faces such as shape of the face, hair, gender to fine-tune a deep CNN for face recognition. Performs better in case of two one-shot face recognition techniques such as exemplar SVM and one-shot similarity kernel. 
- Wu et al. proposed a framework with hybrid classifiers using CNN and Nearest neighbor (NN) model
- Hong et al. proposed a domain adaptation network to solve the One-shot task by generating images in various poses using a 3D face model to train the deep model.
- Zhao et al proposed an enforced softmax that contains optimal dropout, selective attenuation, L2 normalization and model-level optimization which boosted the standard softmax function to produce a better representation for low-shot learning. 

Methodology 
============
1. Siamese NN based approach
2. Deep-feature encoding approach followed by neirest neighbor classification of these encoded features.
3. Combining both
    - Generates features with a ResNet CNN architecture
    - Inputs encoded features to the Siamese network 
    - Siamese network is trained to discriminate between two encoded feature vectores
    - Uses a pretrained DNN (ResNet) as a feature extractor for a pair of input images
    - Energy function is used to ties the twin networks to compute the similarity index
    - Siamese network outputs a similarity score in the 0-1 range between the two encoded feature vectors.

## Siamese Network
- Architectures must be identical
- a
- Sharing weights results in fewer parameters to train and lower tendency of over-fitting
- During training the network learns to discriminate between a pair of images based on the class labels and feature vectors
- Generates probability scores on whether they belong to the same class or not
- **Evaluation**: N way one-shot taks. The network is provided with pairs of images consisting of a reference image and one sample image from each of the n unseen classes at each instance. The label with the highest probability is given to the reference image
### Learning details
- Constant learning rate for all the layers
- Validation metric is calculated every 1000 iterations, choosing the model with the best accuracy
- An early stopping criteria may be included
- Momentum for each layer evolves with a predefined linear slope until it attains a final value of 0.9. It's initialized at 0.5
- Batch size of 8
- L2 regularization penalization
- Weights are initialized using the Glorot uniform initializer. Draws samples from the uniform distribution of [-g,g]. They may be initialized to zero as well
- Siamese Network loss function is computed using a regularized cross entropy loss function. 

## ResNet architecture
Improved VGG-Net which:
- tends to lose generalization once the network increases in depth.
- suffers from the vanishing gradient issue [link](https://machinelearningmastery.com/how-to-fix-vanishing-gradients-using-the-rectified-linear-activation-function/#:~:text=Vanishing%20gradients%20is%20a%20particular,network%20that%20requires%20weight%20updates.) which is an usual problem for deeper networks. ::Solution:: ResNet introduced the 'skip connection' concept. Therefore gradients can flow directly backward from deeper layers by skipping intermediate ones on the backpropagation step
- We may use pruned versions of the original ResNet

### Preprocessing 
- CNN Bounding box generation of a face with 68 Landmark points from an input image
- ResNet is fed with the bounding box information and those 68 activation points inside the face region 
- One may use a pretrained versions of the network to avoid exploiting computational resources

::DATASET::
- Face scrub dataset
- VGG dataset 

### Functionality
- Generates a 128-dimensional encoded feature vector that is fed to the classificator. The network learns weights using _Triplet loss_
- Aim is to: 
  - decrease dissimilarity between the anchor image and positive image
  - increase dissimilarity between anchor and negative image

### Combined Hybrid approach
- Siamese network takes as an input the deep-learned encoded features generated by the ResNet CNN and learns its own set of weights aiming to decrease the cross-entropy loss function.

## Experiments

N-way one-shot task performed on N support classes. Separating one set for evaluating the performance. 

N= 5,10,20,50

### For testing:
Each Support class set S is provided consisting of _'n'_ examples from _'N'_ different unseen classes. 

The algorithm determins which 'N' class each 'n' sample belongs to.

::One may choose to perform a global mean accuracy or individual set accuracy::



