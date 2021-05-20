import cv2

import numpy as np
import imutils
import time
import dlib
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import align_faces as al_f
import equalize_image as eq_i
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb


DETECTOR_NAME = 'R'#"Haar" #"Res10
ALIGN_FACES   = True #False
INPUT_MOVIE = "LuqueNanoDaniel.mp4"

TRACK          = True
TRACKER_REINIT = 5
TRACKERS       = []
ID_dict        = None

if DETECTOR_NAME == 'Haar':
    print("[INFO] loading Haar Detector...")
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
    # Import Movie
    input_movie = cv2.VideoCapture(INPUT_MOVIE)
    
    #For all the video
    while True:
        
        # Grab a single frame of video
        ret, frame = input_movie.read()
            
        # Quit when the input video file ends
        if not ret:
            break
        
        gray          = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        detections    = face_detector.detectMultiScale(gray, 1.3, 5, minSize=(50,50))
        n_detections  = len(detections)
        
else:
    PATH_deploy= 'deploy.prototxt.txt'
    PATH_model= 'res10_300x300_ssd_iter_140000.caffemodel'
    CONFIDENCE= 0.5
    
    # load our serialized model from disk
    print("[INFO] loading RES10 model...")
    net = cv2.dnn.readNetFromCaffe(PATH_deploy, PATH_model)
    
    # Import Movie
    input_movie = cv2.VideoCapture(INPUT_MOVIE)
    #length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    #Initialize variables
    frame_number = 0
    n_valid_detection=0
    tracker_frame_number= 0

    #For all the video
    while True:       
        
        # Grab a single frame of video
        ret, frame = input_movie.read()
        
    
        # Quit when the input video file ends
        if not ret:
            break
        
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), scalefactor=1.0,
                                     size=(300, 300), mean=(104.0, 177.0, 123.0)) 
        
        # obtain detections
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
            
            # Taking care of limits 
            if startX < 0:
                startX = 0
            if startY < 0:
                startY = 0
            if endX > w:
                endX = w
            if endY > h:
                endY = h
            
            
            # draw bounding box of the face given the face detector
            # RED bounding box for face detector
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            
            text_detector = "Detector confidence: {:.2f}%".format(confidence * 100) 
            
            # WRITING ON IMAGE: the confidence of identification
            y_text = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, text_detector, (startX, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
            
            if TRACK == True:
                
                startX_tracker = startX
                startY_tracker = startY
                endX_tracker   = endX
                endY_tracker   = endY
                
                center = np.array([startX+abs(endX-startX)/2, 
                                  startY+abs(endY-startY)/2])
                
                if ID_dict is None:
                    ID_dict = {'ID'   : n_valid_detection, 
                              'Center': center,
                              'StartX': startX,
                              'StartY': startY,
                              'EndX'  : endX,
                              'EndY'  : endY}
                    IDs    = []
                    IDs.append(ID_dict)
                
                
                # For tracking and alignment purposes
                rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                
                # Initializing trackers as an empty list of tracker objects
                if tracker_frame_number==0 and n_valid_detection==0:
                    TRACKERS = []
                    
                
                # Reinitialize frame number for tracking purposes
                if tracker_frame_number>TRACKER_REINIT :
                    tracker_frame_number=0
                    TRACKERS = []
                    
                
                # Initialize tracker
                if tracker_frame_number == 0:
                    
                    dists = []
                    for identity in IDs:
                        a = identity['Center']
                        b = center
                        distance = np.linalg.norm(a-b)
                        dists.append(distance)
                    
                    saveid_index = dists.index(min(dists))
                    saveid       = IDs[saveid_index]['ID']
                    
                    text_tracker = "Tracker init, user:" + str(saveid)
                    
                    print(text_tracker)
                    
                    # Initialize a tracker 
                    tracker = dlib.correlation_tracker()        
                    tracker.start_track(frame, rect)
                    
                    # One tracker per each individual found
                    TRACKERS.append(tracker)
                    
                    # draw GREEN bounding box given the tracker
                    cv2.rectangle(frame, (startX_tracker, startY_tracker), (endX_tracker, endY_tracker), (0, 255, 0), 2)
                     
                        
                    # Write tracker text on frame
                    y_text_tracker = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(frame, text_tracker, (endX-180, y_text_tracker), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
               
                    
                else:
                    print('Tracking...')
                    
                    
                    for tracker in TRACKERS:
                        tracker.update(frame)
                        pos = tracker.get_position()
                        
                        # unpack the position object
                        startX_tracker = int(pos.left())
                        startY_tracker = int(pos.top())
                        endX_tracker   = int(pos.right())
                        endY_tracker   = int(pos.bottom())
                        
                        center_tracker = np.array([startX_tracker+abs(endX_tracker-startX_tracker)/2,
                                                   startY_tracker+abs(endY_tracker-startY_tracker)/2])
                        
                        dists = []
                        for identity in IDs:
                            a = identity['Center']
                            b = center_tracker
                            distance = np.linalg.norm(a-b)
                            print(distance)
                            dists.append(distance)
                        
                        if min(dists) > 100:
                            ID_dict = {'ID': n_valid_detection+1, 
                                       'Center': center_tracker,
                                       'StartX': startX_tracker,
                                       'StartY': startY_tracker,
                                       'EndX':   endX_tracker,
                                       'EndY':   endY_tracker
                                       }
                            IDs.append(ID_dict)
                            
                            # Distance from itself = 0                            
                            dists.append(0)
                        
                        saveid_index = dists.index(min(dists))
                        saveid       = IDs[saveid_index]['ID']
                        
                        # draw bounding box given the tracker
                        cv2.rectangle(frame, (startX_tracker, startY_tracker), (endX_tracker, endY_tracker), (0, 255, 0), 3)
                        
                        
                        text_tracker = "Tracking user:" + str(saveid)
                    
                        print(text_tracker)
                        
                        # Write tracker text on frame
                        y_text_tracker = startY_tracker - 10 if startY_tracker - 10 > 10 else startY_tracker + 10
                        cv2.putText(frame, text_tracker, (endX_tracker-50, y_text_tracker), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                        
                        
                # Slice the frame to obtain the face
                face = frame[startY_tracker:endY_tracker,startX_tracker:endX_tracker,::] 
                plt_face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                
# =============================================================================
#                 if valid_detections>0:
#                     # create subplot and append to ax
#                     ax.append(fig.add_subplot(1, valid_detections, n_valid_detection+1) )
#                     ax[-1].set_title("ID:"+str(saveid))  
#                     plt.imshow(plt_face)
#                     plt.axis('off')
# =============================================================================
                    
                # Save the captured image into the datasets folder
                path="DetectFacesFromVideoImages/person" + str(saveid) +"frame"+str(frame_number)+ ".jpg"
                print(path)
                cv2.imwrite("DetectFacesFromVideoImages/person" + str(saveid) +"frame"+str(frame_number)+ ".jpg", face)
                print("Person ID", i)
            else:
                # Slice the frame to obtain the face
                face = frame[startY:endY,startX:endX,::] 
                plt_face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                
                # Preprocess Image
                # Align given facial landmarks
                _, aligned_face = al_f.face_aligner(frame, rect, show=False)
                
                # Compensate color for histogra equalization
                final_face = face 
                #final_face = eq_i.RGB_hist_equalization(aligned_face)
                
                #final_face = cv2.equalizeHist(aligned_face)
                #cv2.imshow('final face',final_face)
                #key = cv2.waitKey(0)        
            
                # create subplot and append to ax
                ax.append(fig.add_subplot(1, len(IDs), n_valid_detection+1) )
                ax[-1].set_title("ID:"+str(saveid))  
                plt.imshow(plt_face)
                plt.axis('off')
                
                # Save the captured image into the datasets folder
                path="DetectFacesFromVideoImages/person" + str(saveid) +"frame"+str(frame_number)+ ".jpg"
                print(path)
                cv2.imwrite("DetectFacesFromVideoImages/person" + str(saveid) +"frame"+str(frame_number)+ ".jpg", face)
                print("Person ID", i)
            
                # Save the captured image into the datasets folder
                path="DetectFacesFromVideoImages/person" + str(i) +"frame"+str(frame_number)+ ".jpg"
                print(path)
                cv2.imwrite("DetectFacesFromVideoImages/person" + str(i) +"frame"+str(frame_number)+ ".jpg", final_face)
                print("Person ID", i)
            
            n_valid_detection +=1
            print("nVALIDDETECTIONS"+str(n_valid_detection))
        
        # render plot
        #if valid_detections>0:
            #plt.show()
        
        print("Frame for tracker:", tracker_frame_number)
        print("Actual frame", frame_number)
        print("=======================================")
        tracker_frame_number+=1
        frame_number+=1
        cv2.imshow('frame', frame)
        key = cv2.waitKey(0)        
        n_valid_detection = 0
 
        
    
        
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        
    
    input_movie.release()
    cv2.destroyAllWindows()

if ALIGN_FACES== True:
    # create the facial landmark predictor and the face aligner
    face_landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(face_landmarks, desiredFaceWidth=256, desiredFaceHeight=256)


