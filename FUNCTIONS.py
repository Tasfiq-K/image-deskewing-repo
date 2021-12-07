def white_removal(image, thresh_low=245, thresh_high=255, kernel_x=3, kernel_y=3):
    
    """A function to remove white pixels from the given OCT images
       
       Params: image, thresh_low, thresh_high, kernel_x, kernel_y
       Description: image = the source file (should be an image)
                    thresh_low = lower threshold value (default = 245)
                    thresh_high = higher threshold value (default = 255)
                    kernel_x and kernel_y are two interger kernel values for the morphological operation defult is 3 for both"""
       

    import cv2

    oct_image = image.copy()

    gray = cv2.cvtColor(oct_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, thresh_low, thresh_high, cv2.THRESH_BINARY)
    oct_image[thresh == 255] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_x, kernel_y))
    eroded_image = cv2.erode(oct_image, kernel, iterations=1)

    return eroded_image
  

    
def deskew(image):
    """A function to detect the skew angle and rotating the image to deskew
       
       Params      : image
       Descriptions: image = the source file (should be an image) 
       
       performs the white_removal() function on the given image then detects the skewness angle and corrects it"""

    import cv2
    from math import atan2, cos, sin, sqrt, pi
    import numpy as np

    original_image = image.copy()

    image = white_removal(original_image)    # assigns erodded image to image
    
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # convert imgae to binary
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # find all the contours in the thresholded image
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)

        # ignore contours that are too small or too large
        if area < 5000 or 1000000 < area:
            continue
        # cv.minAreaRect returns:
        # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # getting the important parameters from the rectangle
        # (center(x, y), (width, height), angle of rotation)
        center = (int(rect[0][0]), int(rect[0][1]))
        width = int(rect[1][0])
        height = int(rect[1][1])
        angle = rect[2]

        cv2.drawContours(image,[box],0,(0,0,255),2)

        # correcting the angle
        if width < height:
            angle = 90 + angle

            rows, cols, channels = original_image.shape
            M = cv2.getRotationMatrix2D(center, angle, 1)
            dst = cv2.warpAffine(original_image, M, (cols, rows))   # angle corrected image
        
        else:
            angle = angle

            rows, cols, channels = original_image.shape
            M = cv2.getRotationMatrix2D(center, angle, 1)
            dst = cv2.warpAffine(original_image, M, (cols, rows))   # angle corrected image

    return dst, angle


def ROI_deskewed_image(image):

    """Crop the Region Of Interest (ROI) of a deskewed (corrected angle) OCT image
       Param       : image 
       Description : image = should be an deskewed image
       
       Returns: Returns the ROI (cropped) of the OCT image
    """

    import cv2
    from math import atan2, cos, sin, sqrt, pi
    import numpy as np

    original_image = image.copy()

    image = white_removal(original_image)    # assigns erodded image to image
    
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # convert imgae to binary
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # find all the contours in the thresholded image
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)

        # ignore contours that are too small or too large
        if area < 5000 or 1000000 < area:
            continue
        # cv.minAreaRect returns:
        # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # getting the important parameters from the rectangle
        # (center(x, y), (width, height), angle of rotation)
        center = (int(rect[0][0]), int(rect[0][1]))
        width = int(rect[1][0])
        height = int(rect[1][1])
        angle = int(rect[2])

        cv2.drawContours(image,[box],0,(0,0,255),2)

        x_axes = []
        y_axes = []

        for x_coords, y_coords in box:
            x_axes.append(x_coords)
            y_axes.append(y_coords)

        min_x = min([0 if x < 0 else x for x in x_axes])
        max_x = max(x_axes)
        min_y = min([0 if y < 0 else y for y in y_axes])
        max_y = max(y_axes)

        sliced_image = original_image[min_y:max_y+1, min_x:max_x+1]

    return sliced_image

