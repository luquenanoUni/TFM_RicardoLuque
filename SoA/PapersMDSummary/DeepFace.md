Introduction
============

Based on fiducial points in order to warp a detected facial crop to a 3D frontal mode

Facial recognition is based on four main steps
1. Detect
2. Align
3. Represent
4. Classify

::DATASETS::
Trained and validated on the following datasets
- **Labeled faces in the Wild (LFW)**. 
- **Social Face Classification (SFC)**.
- **Youtube Faces (YFT)**. 

Contributions
1. Development of a DNN **Deep Neural Net** architecture to obtain a face representation that generalizes well to other datasets
2. An effective facial alignment based on explicit 3D modeling of faces
3. Achieve near-human-performance on the **LWF** dataset and decrease the error rate on the **YouTube Faces Dataset (YTF)**
- Use of raw images as the underlying representation
- Avoid combining the obtained features with engineered descriptors
- Knowledge transfer can be used after the netweork has been trained on a very large dataset.

::PROS::
Produces a compact image desciptor
::CONS::
The descriptor is sparse
Unconstrained environments still pose a problem


Related work 
=============
## Hand-crafted features.
Employ tens of thousands of image descriptors

## Deep neural nets 
Used for face detection, alignment and verification

## Local Binary Patterns (LBP) 
[link](http://www.scholarpedia.org/article/Local_Binary_Patterns)

Encodes only the relationship between a central point and its neighbours.

State of the art includes using **Local Binary Patterns (LBP) **. They are based on using a texture operator which labels the pixels of an image based on a thresholding of its neighborhood in order to output an output _binary_ number. 
It's highly used in texture recognition and has shown to improve the performance in detection. Combined with **Histograms of oriented gradients (HOG)** features obtains a simple data vector.

#### How does it work:
Threshold the 3x3 neighborhood of each pixel using the center-pixel's value, outputting a binary number. The histogram of these labels can be used as a texture descriptor. It is used jointly with the simple local contrast measure (gray scale variance of the neighbourhood can be used)
It can be extended using **uniform patterns**, which reduces the feature vector's length and implements a rotation-invariant descriptor
**Uniform patterns** are inspired by the fact that some binary patterns occur more commonly in texture than others. *in natural images* A local binary pattern is called uniform if the binary pattern contains at most 2 bitwise transitions from 0 to 1 or vice versa when the bit pattern is traversed circularly. 000000 (0 transitions), 01110000 (2 transitions), 11001111 (2 transitions).

Uniform patterns have a separate label per each uniform pattern, whereas all non-uniform patterns are labeled with a single label.

Labels are given by **P(P-1)+3**, where P is the number of bits in the pattern (e.g. 00011100 8 bits). For example, in a (8,R) neighbourhood a total of 59 labels are considered, 58 for the uniform patterns and 1 for all the non-uniform patterns

Local primitives which are codified by these bins include different types of curved edges, spots, flat areas etc. Low level features.

After obtaining the patterns they are histogrammed and then normalized to add up to 1.


## Spatiotemporal LBP 
[link](https://storage.googleapis.com/plos-corpus-prod/10.1371/journal.pone.0124674/1/pone.0124674.pdf?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=wombat-sa%40plos-prod.iam.gserviceaccount.com%2F20210210%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210210T175132Z&X-Goog-Expires=3600&X-Goog-SignedHeaders=host&X-Goog-Signature=0ac054004d1dd84ed7a1e39daeba0de65711802abaf28398ea8ccbf3de605430e4d3183363b136f0549dd578d663593f200a27835c29bb69809aee69e1c9d141fb0a7c34489cddabec6330c95d16c177fa8c7419880b58c46dcab614fa09b60552040348a3ce3a3fa9dfea2db09d389882930431f29b365f5471af92c66a40135d57b37fe3ad0dc1dfd05c464d6a01d58a1f357189bc7f33dfcdb01b076fe3c7835e90e8da0357ab32acd4cdb396cd1e99479ee38ab56f4ee28ae586fa9a206f6f57a39cd011e21e610dc5a77d58ad9732910e07fa7d1220fab7e1bfa26cd49b12455834f2f154c6bdd6be48249aaa7a3ffd983139c9a1bc8ea79b3b1926d47c)
As an extention of the spatial information in order to obtain a dynamic texture analysis. We use the **Volume Local Binary Patterns (VLBP)**. It consists in looking at dynamic textures as a set of volumes in the (X,Y,T) space. The neighbourhood is defined on a 3D space.

An operator based on co-occurences of local binary patterns on 3 orthogonal planes (XY, XT, YT) called **LBP-TOP** is introduced. It concatenates local binary pattern co-occurence in these directions

- XY represents the appearance information
- XT gives a visual impression of one row changing in time
- YT gives a visual impression of one column changing in time

All three histograms are concatenated into a single one. In which, a dynamic texture is encoded by an appearance and two spatio-temporal co-occurrence statistics

Fine tuning the radius parameters in X,Y and T axes is crucial for analyzing dynamic features.

Classification is performed computing histogram similarities

## Face description using LBP
One should codify the texture information while retaining their locations. Ways to achieve it:

1. Use LBP texture descriptors to build several local descriptions of the face and combine them into a global description. ::Pros: More robust against variations in pose or illumination:: 
  - The basic methodology:
    1. The facial image is divided into local regions. ::Regions:: do not need to be the same shape or size, they do not need to cover the whole image, partially overlapping regions may be also considered.
    2. LBP texture descriptors are extracted from each region independently
    3. Descriptors are concatenated to form a global description of the face.

The LBP layers provide a description of the faces in three different levels.
1. Information about the pattern on a Pixel level
2. Information on a regional level: by summing over all labels within the region (histograms)
3. Information on a global level: achieved by concatenating histograms

### Drawbacks

- Very sensitive to image noise
- Unable to capture macrostructure information
- High feature dimensionality
- Encodes only the relationship between a central point and its neighbours

## Joint Bayesian Model (JBM)
Adapting the JBM using images from subjects to the LFW domain.
Using [distance Metric learning](http://contrib.scikit-learn.org/metric-learn/introduction.html#:~:text=Distance%20metric%20learning%20(or%20simply,%2C%20clustering%2C%20information%20retrieval).), which aims to automatically construct a _task-specific_ metric that can be used to assess unsupervised or weakly supervised learning

Representation
==============

## Face Aligmnent 
1. Employing an analytical 3D model of the face
2. Searching for similar fiducial-points configurations from an external dataset 
3. Unsupervised methods that find a similarity transformation for the pixels

Using **LBP** hisrograms as image descriptors. 
- Transform the image using an induced similarity matrix and run the fiducial detector on this new feature space
### Alignment process
1. **2D alignment** Detect 6 fiducial points centered at eyes - tipo of the nose - mouth locations. Used to approximate scale, rotation and translation. Obtaining the final 2D similarity transformation
2. Generate a 2D aligned crop (boosts recognition accuracy) ::CON:: fails to compensate out-of-plane rotation
3. **3D alignment** ::Attention here:: Generate a 3D-aligned version of the crop in order to deal with out of plane rotations. Localize 67 fiducial points. Use a 3D shape model and register a 3D affine camera, which warp the 2D crop to the image plane of the 3D shape. ::Manually:: place 67 anchor points on the 3D shape. Fit a 3D-to-2D camera **P**. ::Detected points on the countour of the face tend to be more noisy as the estimated location is largely influenced by the depth with respect to the camera angle::
- **Frontalization** In order to reduce corruption. Add _r_ residuals to x-y components of each reference fiducial point. This reduces distortion and increases discriminative factors
### Representation
4. Learn a generic representation of facial images through a **DNN Architecture and Traning**. It's trained on multi-class face recognition task. To classify the identity of a face image. ::Inputs 152x152:: 
  - ::PREPROCESSING:: C1 (Conv), M2 (MaxPool), C3 (Conv). C1 and C3 are used to extract low-level features like edges and texture. M2 increases robustness to local translations. Excessive pooling leads to infromation loss of the position of facial structure and micro-textures
  - L4, L5, L6 are [locally connected layers](https://prateekvjoshi.com/2016/04/12/understanding-locally-connected-layers-in-convolutional-neural-networks/) in which different regions of an aligned image have different local statistics (no weight sharing). Computational burden isn't really increased, but there are more parameters to be trained. Hardly any statistical sharing between large patches in aligned faces.
  - F7 and F8 are fully connected. Are able to capture correlations between features captured in distant part of the face images. The ouput of F7 is used as our raw face representation feature vector i.e. input to the classifier 
  - ::Different from LBP:: LBP pools local descriptors and uses this as input to a classifier.
  - F8's output is fed to a K-way softmax in order to produce a distribution over the class labels. We minimize the cross-entropy loss for each training sample. 75% of feature components in the topmost layers are zero due to the use of ::RELU:: ::THIS COULD BE INTERESTING TO CHECK, MAYBE LEAKY RELU or other activations functions DO SOMETHING NICER?:: It's applied after every conv, locally and fully connected layer (minus the last one). Sparsity is also encouraged by using dropout on the 1st fully-connected layer. 
### Normalization
Features are normalized to 0-1 in order to reduce the sensitivity to illumination changes. ::RELU:: does not provide invariance to re-scaling of image intensities (some of them are sent to 0)

::CONS::
- Not the best one under unsupervised environments
- Needs fine tuning given the target domain
- Not easily suitable for generalization in large datasets
- Bad generalization if the model has been trained in a relatively small dataset

Performance 
===========
- Using the weighted $x^2$ similarity, where the weight parameters are learned using SVMs.
- Siamese network: The face recognition network is replicated twice and the features are used to predict whether the two input images belong to the same person. Increases computation to twice. Uses cross entropy loss and back propagation of the error
    1. Taking the absolute difference between features
    2. A FCN that maps into logistic (same/not) output. 
    3. Training is performed only on 2 topmost layers to avoid overfitting
Up to 97.25% accuracy



Further reading
===============
- ::Liu et al. (2016) Median Robust Extended Local Binary pattern **MRELBP** ::
- Automatic fiducial points detection for facial expressions using scale invariant feature [link](https://ieeexplore.ieee.org/document/5293308)
- ::Efficient Spatio-Temporal Local Binary Patterns for Spontaneous Facial Micro-Expression Recognition:: [link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0124674)
- Zhang et al. (2005) LBP feature extraction by filtering a facial image with different scales and orientations of Gabor filters.
- Hadid and Pietikäinen (2009) Spatiotempral LBPs for face and gender recognition from video sequences.
- Zhao et al. (2009) LBP-TOP used for visual speech recognition.
- Liao et al. (2009) Using dominant local binary patterns to improve recognition accuracy 
- Heikkilä at al. (2009) using SIFT and LBP descriptors. It uses **center-symmetric local binary patterns (CS-LBP)** to replace the gradient operator used by SIFT.
- Tan et al. (2007), Wang et al. (2009) Combination of LBPs and Gabor features