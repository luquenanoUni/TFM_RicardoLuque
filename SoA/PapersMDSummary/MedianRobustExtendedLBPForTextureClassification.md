Introduction
============

# LBP 
Initially proposed in 1994, by Wang and He \cite{}, LBP introduced a model of texture analysis based on a texture unit, where a texture can be characterized by its spectrum. Further modified by Ojala et al \cite{} in 1996, a two-level and more robust version of the texture unit known as Local binary patterns is introduced. Initially encoding a 3x3 neighbourhood from each center pixel. Each value of the eight pixels are added up until one obtains a texture unit. 

Although it was initially built for texture classification and matching, it was further extended for facial description

# Types

MRELBP compares regional image medians rather than raw image intensities.
A multiscale LBP type descriptor is computed by comparing image medians which can capture micro and macrostructure texture information.

::Advantages:: 
- high performance
- robust to gray scale variations, rotation and noise
- High classification scores of 99.82, 99.38 and 99.77 on Outex test suites. 
- Highly robust to Gaussian noise, Gaussian blur, salt-and-pepper noise and random pixel corruption

Extraction of powerful texture features plays an important role since poor features will fail to achieve good recognition results even if using a good classifier.
Extracting powerful feature features lies in balancing
- High-quality descriptors
  - Deal with distinctiveness and robustness tradeoff
- Low computational complexity 

It's a solution of LBP & variants which suffer from robustness as they have minimal tolerance to image blur and noise corruption.

::Main contributions::
- Novel sampling scheme which can encapsulate micro and macrosture information
- Combining local means with the sampling scheme proves to be a powerful texture feature
- Gray scale and rotation invariance
- No pretraining or parameter tuning
- Discriminativeness and noise robustness 

Extentions of LBP include **LBPRI** (rotation invariant) which performs an i-step circular bit-wise right shift on x. Keeping only those rotaionally-unique patterns reduces feature dimensionality.

Local binary pattern uniform descriptor LBPRIU2 reduces the dimansionality to p+2

## Extended Local Binary pattern (ELBP)
Is designed to encode distinctive spatial relationships in a local region and therefore contains more spatial information.

Consists of exploring information from
- *ELBP_CI*: the intensity of the centers
- *ELBP_NI*: neighbouring pixels 
- *ELBP_RD*: Radial differences

*ELBP_CI*: The center pixel is thresholded against 
$\beta$, the mean of the whole image.

*ELBP_NI*: uses the average of neighbouring pixels' intensities to generate the binary patttern. We have a local mean given by the neighbouring pixels.

*ELBP_RD*: uses pixel differences in radial directions

## LBP variants
Aim to explore anisotropic information (not designed for rotation invariance)
*GLTP*: Geometrical local Texture Pattterns explores intensity changes on oriented neighborhoods. 
*Different topologies*: Using different neighbourhood topologies (circle, ellipse, parabola, hyperbola and archimedean spiral).
*LQP*: Local Quantized patterns, a selection of possible geometries are evaluated.

### Increasing discriminative power: 
Three primary strategies to improve discriminative power
1. Reclassifying the original LBP patterns to form more discriminative clusters
2. Exploring co-occurrences
3. Combining other texture descriptors

*Haming LBP*: regroups nonuniform patterns based on Hamming distance instead of collecting them into a single bin.
*Learning discriminative rotation invariant patterns*
*Introducing Pairwise Rotation Invariant Cooccurrence LBP* **PRICoLBP**: makes use of the co-ocurrences of pairs of LBPs at certain relative displacement
*Multiscale Joint LBP*: consideres co-occurrences of LBPs, but in different scales. 
*Local contrast descriptor VAR to combine with LBP* 
**VAR is a rotation invariante measure of the local variance.**
*Combination of Gabor filters and LBP* Liao et al.
*LBP Fourier histogram (LBPHF)* Ahonen eet al. achieves global rotation invariance 
*Completed LBP (CLBP)* the local differences are decomposed into signs and magnitudes Guo et al
*LBP+Local Neighbouring intensity relationship (LNIRP)* based on a sampling structure which combines pixel and patch to mimic the retinal sampling gripd. 
AD-LBP ::SEARCH::

### Enhancing Noise robustness
*Soft LBP (SLBP)* enhances robustness by incorporating fuzzy membership in the representation of local texture primitives
*Fuzzy LBP (FLBP)* allows multiple LBPs to be generated at each pixel position. 
::Both are computationally complex::
*Noise Resistant LBP (NRLBP)* ::Less computationally complex:: 
*Local Ternary Patterns (LTP)*: 
::Pros::
- More robust to noise than LBP
::Cons::
- Not strictly invariant to gray scale changes
- Fine tuning of the additional threholds is not simple
*Dominant LBP (DLBP* learns most frequently occurred patterns to capture descriptive textural information
::Cons::
- Requires pretraining
*Median Binary Pattern (MBP)* local binary patterns are determined by a localized threholding against the local median
*Local Phase Quantization (LPQ)*: robustness to image blur
*Noise Tolerant LBP (NTLBP)* uses a circular majority voting filter and a new encoding strategy that regroups non-uniform LBP patterns
*Robust LBP (RLBP)* changes the coding of LBP.

Robust Extended Local Binary Pattern (RELBP)
====================
## Strategies
###  Replacing individual pixel intensities 
Replacing individual pixel intensities at a point with some representation over a region.


### ::BRIEF, BRISK & FREAK:: 
For which a binary descriptor vector is obtained by comparing number of pairs of pixels after applying a Gaussian smoothing, to fight noise sensitivity.
- Based on keypoint detection
- Characterization of each keypoint
::BRISK & FREAK:: depend on the detection of local regions of interest and estimation of dominant orientations (Similar to SIFT)

#### Drawbacks: 
- are outperformed by dense approaches

Modification of the ELBP descriptor so that individual pixel intensities are replaced by a filter response $\sigma$

RELBP is achieved through a joint histogramming of *RELBP_CI*, *RELBP_NI* and *RELBP_NI*

## RELBP choices
- Gaussian RELBP: sampling after Gaussian smoothing
- Averaging RELBP: regional mean
- Median RELBP: regional median

::Gaussian and Averaging RELBP:: perform spatial averaging and therefore noise reduction, however the methods are linear and have limited robustness.

## Encoding Scheme
Rotation invariant uniform riu2 has become the standard, it classifies all of the uniform LBPs into p+1 rotation invariant groups and places all the remaining nonuniform patterns into one single group.

**_UNIFORM LBPS do not necessarily represent the most significant pattern features for certain classes of textured images, therefore grouping patterns into one group may result in information loss_**
1. RELBP_riu2: Rotation invariant uniform encoding scheme. (P+1)
2. RELBP_ri: P+2
3. RELBP_ham: Some nonuniform patterns are reclassified by minimizing a hamming distance
4. RELBP_Faith. All nonuniform patterns with four bitwise transitions (U) are classified based on the number of ones in the pattern, and the nonuniform patterns with U > 4 are grouped by U value
5. RELBP_count: Patterns are grouped into p+1 different groups based on counting the number of ones
6. ::RELBP_num:: 
  - dividing all LBPs into uniform and nonuniform according to the uniformity measure. 
  - Then patterns are divided into p+1 rotation invariant groups.
  - Group the nonuniform pattern into p-3 different groups based on th enumber of ones in the pattern


## MultiScale analysis and classification
By altering r and p we can make operators for any quantization of the angular space for any spatial resolution. A multiresolution analysis is achieved by concatenating binary histograms from multiple resolution into a single histogram
- No independence from texture features on different scales
- The estimation of large joint probabilities is not feasible due to computational complexity of large multidimensional histograms.
- ::Instead the histogram feature is generated as the concatenation over multiple scales::

::DAISY:: built for dense matching. Dense image matching takes an alternative approach to obtain a corresponding point for almost every pixel in the image. Rather than searching the entire image for features, it will compare two overlapping images row by row. Essentially, this reduces the problem to a much simpler one-dimensional search. 

::BRISK AND FREAK:: image matching

Experimental evaluation
=======================
The classification is performed via the Nearest Neighbor Classifier applied to the MRELBP histogram feature vectures and using the $X^2$ distance metric. A SVM is also used.

**Datasets**
Outex
CUReT
UMD
KTHTIPS2b
ALOT

::PROS::
1. Robustness to gray scale and rotation variations
2. Robustness to random noise corruption
  - Gaussian noise 
  - image blurring
  - salt and pepper noise
  - random pixel corruption
3. Robustness to more complex environment changes on **CUReT, UMD, KTHTIPS2B and ALOT** databases
  - Viewpoint variations
  - scaling
  - illumination
  - rotation 

::DRAWBACKS::

Still not the best robustness to random corrupted pixels 
Results are highly dependant on fine tunning 
