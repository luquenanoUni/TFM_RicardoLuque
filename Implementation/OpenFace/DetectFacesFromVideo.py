import cv2

import numpy as np
import imutils
import time
import dlib
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import align_faces as al_f
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

PATH_deploy= 'deploy.prototxt.txt'
PATH_model= 'res10_300x300_ssd_iter_140000.caffemodel'
CONFIDENCE= 0.5

# create the facial landmark predictor and the face aligner
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)


input_movie = cv2.VideoCapture("LuqueNanoDaniel.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(PATH_deploy, PATH_model)

#Initialize variables
frame_number = 0
n_valid_detection=0
tracker_frame_number= 0
reinit = 5

trackers       = []
#For all the video
while True:
    
    # Reinitialize frame number for tracking purposes
    if tracker_frame_number>reinit:
        tracker_frame_number=0
        trackers = []

    
    # Grab a single frame of video
    ret, frame = input_movie.read()
    

    # Quit when the input video file ends
    if not ret:
        break

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    
    # pass the blob through the network and obtain the detections and
	# predictions
    net.setInput(blob)
    detections = net.forward()
    
    # Obtain initial detections without confidence filtering
    n_detections = detections.shape[2]
    
    # Obtain number of valid detections
    valid_detections=np.sum(detections[0,0,:,2]>CONFIDENCE)
    
    # For plotting
    fig = plt.figure(figsize=(5, 5))

    # for the subplots
    ax = []
    
    # loop over the detections
    for i in range(n_detections):
        
              
        # extract the confidence associated with the prediction
        confidence = detections[0, 0, i, 2]
             
        # filter out weak detections
        if confidence < CONFIDENCE:
            continue
        
        
        # compute the (x, y)-coordinates of the bounding box
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        if startX < 0:
            startX = 0
        if startY < 0:
            startY = 0
        if endX > w:
            endX = w
        if endY > h:
            endY = h
        
        # draw bounding box of the face given the face detector
        # R channel for face detector
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        
        text_detector = "Detector confidence: {:.2f}%".format(confidence * 100) 
        
        # show the confidence of identification
        y_text = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, text_detector, (startX, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
        
        # Initialize tracker
        if tracker_frame_number == 0:
            
            print('(Re)Initializing Tracker')
            
            # For tracking purposes
            rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                        
            # Initialize a tracker 
            tracker = dlib.correlation_tracker()        
            tracker.start_track(frame, rect)
            
            # One tracker per each individual found
            trackers.append(tracker)
            
            # draw bounding box given the tracker
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 3)
            
            text_tracker = "Tracker (Re)initialization, user:" + str(i)
            #text_tracker = "Initializing Tracker: {:i}%".format(n) 

                
            # Write tracker text on frame
            y_text_tracker = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, text_tracker, (endX-180, y_text_tracker), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
       
            
        else:
            print('Tracking')
            
            # HERE I NEED TO ADD A FLAG
            tracker_flag = True
            
            if tracker_flag == True:
                for n, tracker in enumerate(trackers):
                    print('Person being tracked',n)
                    tracker.update(frame)
                    pos = tracker.get_position()
                    
                    # unpack the position object
                    startX_tracker = int(pos.left())
                    startY_tracker = int(pos.top())
                    endX_tracker   = int(pos.right())
                    endY_tracker   = int(pos.bottom())
        
                    # draw bounding box given the tracker
                    cv2.rectangle(frame, (startX_tracker, startY_tracker), (endX_tracker, endY_tracker), (0, 255, 0), 3)
                    
                    text_tracker = "Tracker" 
                    #text_tracker = "Tracker:"+str(n)
                    
                    # Write tracker text on frame
                    y_text_tracker = startY_tracker - 10 if startY_tracker - 10 > 10 else startY_tracker + 10
                    cv2.putText(frame, text_tracker, (endX_tracker-50, y_text_tracker), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    print("Writing in position:" +str(y_text_tracker))
                
                   
        
        # Slice the frame to obtain the face
        face = frame[startY:endY,startX:endX,::] 
        plt_face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        
        aligned_face = al_f.face_aligner(frame, face,rect,show=True)

        
        # create subplot and append to ax
        ax.append( fig.add_subplot(1, valid_detections, i+1) )
        ax[-1].set_title("ID:"+str(i))  
        plt.imshow(plt_face)
        plt.axis('off')
        
        # Save the captured image into the datasets folder
        path="DetectFacesFromVideoImages/person" + str(i) +"frame"+str(frame_number)+ ".jpg"
        print(path)
        cv2.imwrite("DetectFacesFromVideoImages/person" + str(i) +"frame"+str(frame_number)+ ".jpg", face)
        print("Person ID", i)
        
        n_valid_detection +=1 

    # render plot
    plt.show()
    
    print("Frame for tracker:", tracker_frame_number)
    print("Actual frame", frame_number)
    print("=======================================")
    tracker_frame_number+=1
    frame_number+=1
    cv2.imshow('frame', frame)
    key = cv2.waitKey(0)         
    

    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    

input_movie.release()
cv2.destroyAllWindows()