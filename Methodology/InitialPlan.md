Methodology for implementation
==============================
1. Define a CNN architecture based on ResNet/DeepFace architecture for feature extraction
2. Fetch the LFW dataset and build a reduced version of it
3. Deal with class imbalance using Guo's class promotion loss term to promote underrepresented classes [link](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/07/one-shot-face.pdf)
4. Deal with class imbalance using SMOTE and Balance Cascade approaches. [link](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w48/Zhang_Class-Balanced_Training_for_Deep_Face_Recognition_CVPRW_2020_paper.pdf)
5. Implement the given architecture(s)
6. Transfer learning from first layers 
7. **NOT NOW** Train both architectures on the three balanced subsets using one of the CNN architectures
8. Review the accuracy scores, ROC curves and time performances in classification. Times of training and classification. Metrics in general.
9. **NOT NOW** Select the best data subset given the metrics.
10. Train the network on the second architecture 
11. Review performance metrics as well as times of training and classification.
12. Duplicate CNN and create siamese network
13. Add extra layer after obtaining the feature vectors to enforce similarity among samples belonging to same class using a cross-entropy approach. Try performance using both -duplet and triplet loss
14. Check metrics given a handful of samples to verify the behavior of the network. 

Make it more shallow