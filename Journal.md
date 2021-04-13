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
  
