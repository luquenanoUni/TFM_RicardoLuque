import cv2

def RGB_hist_equalization(img):
    # Convert to YUV color space
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        
    # convert the YUV image back to RGB format
    eq_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    return eq_img
        