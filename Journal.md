Journal
=======
## Wednesday 07th April:
  - Extract faces that are detected and save them for few shot learning purposes. **Problems: ** Not able to perform it when more than one face is present in the scene. Problems with color channels. 
  - Report writing
  - 
## Thursday 08th April:
Problem with color channels has been solved. Several faces can be extracted now. **Problems:** Wrong labelling of the faces given the lack of tracking. Plotting problems with matplotlib, wrong management of axis.  

## Friday 09th April: 
Tracking is introduced with correlation tracker. **Problems:** Only one bounding box is drawn although several trackers (one per each individual) are defined

## Monday 12th April: 
Solving tracking problems. Still not perfectly done since bounding boxes are tracking the objects, yet identification per each user is not happening. 

## Tuesday 13th April: 
Solving user saving and plotting problems. Different people are being identified, yet IDs are still problematic given each specific user. 
Tengo que asignarle a cada tracker un rostro de cada individuo sin que se repita. 

## Wednesday, April 14th  
Still problems with ID, yet the repetition problem of the same individual being identified twice has been solved. Still each user isn't tracked independently.

## Thursday, April 15th 
Implementing Aligning algorithm based on Facial landmarks


## Friday, April 16th 
Drawing diagrams for the whole system. Organizing some scripts and locations for merging pieces of code together. Writing down a bit of the implementation on Overleaf. Finishing the alignment algorithm merging into the main code

## Monday, April 19th
Trying out contrastNet: Cnn-Based Detection of Generic Contrast Adjustment with Jpeg Post-Processing [link](https://github.com/andreacos/ContrastNet). Didn't work due to lack of parametrization given the vagueness of the paper. A simple histogram manipulation is performed, so that all images colors are equalized.


## Tuesday, April 20th 
Facial recognition works for contrastive loss, however similar looking features usch as presence of glasses, similar facial expressions and/or skin tone make it hard for the system to achieve a correct person re-Identification. 

## Wednesday, April 21st
Trying to merge the pretrained model to the existing siamese network, so that a different backbone (aiming to improve performance) is used. Before this models should be trained and weights saved so that they're called from presaved weights instead.

## Thursday, April 22nd
Fetching the trained models reaching an accuracy of 0.84 on the LFW dataset. 
**MODELS ARE SAVED LOCALLY, THEY CAN'T BE PUSED, WILL BE UPLOADED TO BOX**

## Friday, April 23rd.
- Implementation of One shot learning with triplet loss. We are not building a direct classifier, we are building a similarity comparer. Problems within the network. How long should your embedding be? The idea is that your embedding must be able to contain a “good enough” representation of your class to differentiate it from the others. The embedding length is a new hyper-parameter of the problem.
- Building functions for triplets. Choosing the right triplets. Splitting data into new buckets.

##  Monday, April 26th 
Solving problems with triplet format, size and preprocessing 

Entrenar la red y añadir los nuevos samples a la red.

Choosing the right triplets [link](https://omoindrot.github.io/triplet-loss) 

## Tuesday, April 27th
Solving problems with layers, adding layers, substracting others, editing models and saving them

## Wednesday, April 28th
Implementing a layer class to add on top of any pre-existing network in order to train the weights of the networks given a new loss function. 

## Thursday, April 29th.
Implementing loss function and adding embedded vectors as input to the pre-existing network. 

## Friday, April 30th
Reducing network parameters for optimal training under Colab restrictions of RAM. Fetching pretrained models from ResNet initially trained on 

## Moday, May 3rd.
Training the network. Writing of the documentation.