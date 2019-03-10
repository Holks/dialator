import cv2
import numpy as np
from matplotlib import pyplot as plt
import operator
import os

def get_distance_from_line(line, point):
    # line slope
    a = line['a']
    # line intercept
    b = line['b']
    c = line['c']
    x = point[0]
    y = point[1]
    dist = abs(-a*x + b*y - c) / np.sqrt(np.square(a) + np.square(b))
    try:
        closest=(
            int((b*(b*x+a*y)-a*c)/(np.square(a) + np.square(b))),
            int((-a*(-b*x-a*y)+b*c)/(np.square(a) + np.square(b))))
    except:
        closest = (None, None)
    return dist, closest

class dial_indicator_KI():
    def __init__(self):
        self.center = None
        self.radius = None
        self.contours = None

    def get_center_w_hough(self, img):
        cimg = img.copy()
        cimg = cv2.medianBlur(cimg,5)
        h,w = img.shape
        self.h  = h
        self.w = w
        maxRadius = int(h/2)
        minRadius = int(maxRadius*0.5)
        print(h,w ,maxRadius, minRadius)
        edges = cv2.Canny(cimg,30,100,apertureSize = 3)
        circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,150,param1=200,
                    param2=100,minRadius=minRadius,maxRadius=maxRadius)
        try:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(cimg,(i[0],i[1]),i[2],(255,0,0),int(np.ceil(h/140)))
                # draw the center of the circle
                self.center = (i[0],i[1])
                cv2.circle(cimg,self.center,1,(0,0,255),3)

        except:
            pass
        return cimg

    def get_center_w_max_child_contour(self, img):
        # apply blurring
        self.blur = cv2.GaussianBlur(img,(9,9),0)
        # adaptive thresholding
        self.thresh = cv2.adaptiveThreshold(self.blur,255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,3)
        # find contours and their hierarchy
        contours, self.hierarchy = cv2.findContours(self.thresh,
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        self.contours = [(i, cv2.contourArea(cnt), cnt) for i,cnt \
            in enumerate(contours)]
        max_idx, max_val, max_cnt = max(self.contours,
            key=operator.itemgetter(1))
        children = [(i, cv2.contourArea(child), child) for i,child \
            in enumerate(contours) if self.hierarchy[0][i][3] == max_idx]

        max_child_idx, max_child_area, max_child_cnt = max(children,
            key=operator.itemgetter(1))
        (x,y), radius = cv2.minEnclosingCircle(max_child_cnt)
        self.center = (int(x),int(y))
        self.radius = int(radius*0.9)
        return self.center, self.radius

    def get_masked_img(self,img):
        if not hasattr(self, 'self.mask'):
            mask = np.zeros_like(img)
            self.mask = cv2.circle(mask,self.center,self.radius,(255,0,0),-1)
        cimg = cv2.bitwise_and(img,self.mask)
        return cimg
    def normalised_img(self, img):
        if hasattr(self, 'self.mask'):
            roihist,roihist,0,255,cv2.NORM_MINMAX
            norm_img = cv2.normalize(img, None, alpha=0, beta=255,
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, mask=self.mask)
        else:
            norm_img = cv2.normalize(img, None, alpha=0, beta=255,
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return norm_img

    def get_ticks(self):
        pass

    def get_dial_value(self):
        pass

    def get_avg_mask(self):
        pass

    def get_lines(self, img, mask=None):
        """
        edges = cv2.Canny(mask_inside,100,200)

        minLineLength = 100
        maxLineGap = 1
        lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
        if lines:
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(mask_inside,(x1,y1),(x2,y2),(255,255,0),2)"""
        if mask is not None:
            img = cv2.bitwise_and(img,mask)
        self.edges = cv2.Canny(img,30,100,apertureSize = 3)

        minLineLength = 20
        maxLineGap = 2
        lines = cv2.HoughLinesP(self.edges,1,np.pi/180,40,minLineLength,maxLineGap)
        center_corrected = []
        if lines is not None:
            for line in lines:
                for x1,y1,x2,y2 in line:
                    a = (y2-y1)/(x2-x1)
                    b = 1
                    c = y1 - a*x1
                    line = {
                    'a': a,
                    'b': b,
                    'c': c}
                    dist, closest = get_distance_from_line(line, self.center)
                    if dist < 5:
                        print(self.center, closest)
                        center_corrected.append(closest)
                        cv2.line(img,(x1,y1),closest,(255,255,0),1)
        try:
            (x,y),radius = cv2.minEnclosingCircle(np.asarray(center_corrected))
            self.center = (int(x),int(y))
        except:
            pass
        """
        lines = cv2.HoughLines(self.edges,1,np.pi/180,20)

        if hasattr(self,'lines'):
            for line in lines:
                for rho,theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)"""
        return img
    def get_dial_axis_w_hough(self, img, mask=None):
        cimg = img.copy()
        cimg = cv2.medianBlur(cimg,5)
        mask_radius = int(self.radius*0.2)
        if mask is not None:
            img = cv2.bitwise_and(cimg,mask)
        else:
            mask = np.zeros_like(raw_img)
            mask = cv2.circle(mask,self.center,mask_radius,(255,0,0),-1)
        cimg = cv2.bitwise_and(cimg,mask)
        edges = cv2.Canny(cimg,75,150,apertureSize = 5)
        circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,50,param1=75,
                    param2=30,minRadius=int(mask_radius*0.1),maxRadius=int(mask_radius*0.6))
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                #cv2.circle(cimg,(i[0],i[1]),i[2],(255,0,0),1)
                self.center = (i[0],i[1])
                cv2.drawMarker(cimg, self.center, (255,0,0), cv2.MARKER_CROSS, 50,1)
        return cimg

    """
    r1 = int(radius*0.85)
    r2 = int(radius*0.75)

    mask_full = np.zeros_like(img)
    mask_inside = np.zeros_like(img)
    mask_full = cv2.circle(mask_full,center,r1,(255,0,0),-1)
    mask_inside = cv2.circle(mask_inside,center,r2,(255,0,0),-1)
    # draw white circles to get full contours
    kernel = np.ones((2,2),np.uint8)
    #erosion = cv2.erode(thresh,kernel,iterations = 1)
    erosion = cv2.bitwise_and(thresh,thresh,mask = mask_full)
    erosion = cv2.bitwise_and(erosion,erosion,mask = cv2.bitwise_not(mask_inside))

    erosion = cv2.circle(erosion,center,r1,(255,0,0),1)
    erosion = cv2.circle(erosion,center,r2,(255,0,0),20)
    contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)[-2:]
    ticks =  [cv2.contourArea(cnt) for cnt in contours]
    mask_full = cv2.bitwise_and(img,mask_full)

    mask_inside = cv2.bitwise_and(img,mask_inside)
    avg_tick = np.median(ticks)*0.7
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > avg_tick:
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            center = (int(x),int(y))
            cv2.drawMarker(img, center, (255,0,0), cv2.MARKER_SQUARE, 2,2)
    """

if __name__=="__main__":
    #root_dir = input("input images directory\n").strip('"')
    root_dir = '/home/holger/Pictures/Kellindikaator'
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in files:
            filename = os.path.join(root_dir, name)
            print(filename)
            if '.jpg' in name:
                img = cv2.imread(filename, 0)
                dial = dial_indicator_KI()
                raw_img = img.copy()
                center, radius = dial.get_center_w_max_child_contour(img)
                center_img = cv2.circle(img,center,radius,(255,0,0),5)
                thresh = dial.thresh

                masked_img = dial.get_masked_img(raw_img)

                mask_inside = np.zeros_like(raw_img)
                mask_inside = cv2.circle(mask_inside,center,int(radius*0.7),(255,0,0),-1)
                mask_inside = cv2.bitwise_not(mask_inside)

                norm_img = dial.normalised_img(masked_img)
                norm_img = cv2.convertScaleAbs(norm_img)
                hough_circles = dial.get_center_w_hough(raw_img)
                hough_lines = dial.get_lines(norm_img, mask=mask_inside)
                hough_dial_axis = dial.get_dial_axis_w_hough(raw_img)
                edges = dial.edges
                cv2.drawMarker(hough_circles, dial.center, (255,0,0), cv2.MARKER_CROSS, 50,1)
                images = [raw_img, center_img, edges, hough_circles, dial.edges, hough_dial_axis]
                titles = []
                plt.subplot(2,3,1),plt.imshow(images[0],'gray')
                plt.subplot(2,3,2),plt.imshow(images[1],'gray')
                plt.subplot(2,3,3),plt.imshow(images[2])
                plt.subplot(2,3,4),plt.imshow(images[3],'gray')
                plt.subplot(2,3,5),plt.imshow(images[4],'gray')
                plt.subplot(2,3,6),plt.imshow(images[5])
                plt.subplots_adjust(wspace=0, hspace=0, left=0.05, right=0.95,
                    bottom=0.05,top=0.95)
                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                plt.show()
            else:
                print("not an image")
