# USAGE
# python align_faces_parsed.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/01.jpg

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2


# create the facial landmark predictor and the face aligner
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)

def face_aligner(frame, rect, show):
    # resize input image  and convert it to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # extract the ROI of the *original* face, then align the face
	# using facial landmarks

    (x, y, w, h) = rect_to_bb(rect)
    #faceOrig = frame[y:y + h, x:x + w,::]
    faceOrig = imutils.resize(frame[y:y + h, x:x + w,::], width=256)
    faceAligned = fa.align(frame, gray, rect)

	# display the output images
    if show==True:
        cv2.imshow("Original", faceOrig)
        cv2.imshow("Aligned", faceAligned)
        cv2.waitKey(0)
    
    return faceOrig, faceAligned