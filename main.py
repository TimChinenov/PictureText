#Text image interpreter
#By Tim Chinenov
import Letter
#import statistics
import cv2
import numpy as np
from matplotlib import pyplot as plt


#function finds the corners given the top,bottom,left,and right
#maximum pixels
def findCorners(bound):
    c1 = [bound[3][0],bound[0][1]]
    c2 = [bound[1][0],bound[0][1]]
    c3 = [bound[1][0],bound[2][1]]
    c4 = [bound[3][0],bound[2][1]]
    return [c1,c2,c3,c4]

#function finds the minimization of the weighted within-class variance
#this algorithm is adapted from:
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
def findThresh(data):
    Binsize = 50
    #find density and bounds of histogram of data
    density,bds = np.histogram(data,bins=Binsize)
    #normalize the histogram values
    norm_dens = (density)/float(sum(density))
    #find discrete cumulative density function
    cum_dist = norm_dens.cumsum()
    #initial values to be overwritten
    fn_min = np.inf
    thresh = -1
    bounds = range(1,Binsize)
    #begin minimization routine
    for itr in range(0,Binsize):
        if(itr == Binsize-1):
            break;
        p1 = np.asarray(norm_dens[0:itr])
        p2 = np.asarray(norm_dens[itr+1:])
        q1 = cum_dist[itr]
        q2 = cum_dist[-1] - q1
        b1 = np.asarray(bounds[0:itr])
        b2 = np.asarray(bounds[itr:])
        #find means
        m1 = np.sum(p1*b1)/q1
        m2 = np.sum(p2*b2)/q2
        #find variance
        v1 = np.sum(((b1-m1)**2)*p1)/q1
        v2 = np.sum(((b2-m2)**2)*p2)/q2

        #calculate minimization function and replace values
        #if appropriate
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = itr

    return thresh,bds[thresh]

def dist(P1,P2):
    return np.sqrt((P1[0]-P2[0])**2+(P1[1]-P2[1])**2)

#function takes two rectangles of corners and combines them into a single
#rectangle
def mergeBoxes(c1,c2):
    newRect = []
    #find new corner for the top left
    cx = min(c1[0][0],c2[0][0])
    cy = min(c1[0][1],c2[0][1])
    newRect.append([cx,cy])
    #find new corner for the top right
    cx = max(c1[1][0],c2[1][0])
    cy = min(c1[1][1],c2[1][1])
    newRect.append([cx,cy])
    #find new corner for bottm right
    cx = max(c1[2][0],c2[2][0])
    cy = max(c1[2][1],c2[2][1])
    newRect.append([cx,cy])
    #find new corner for bottm left
    cx = min(c1[3][0],c2[3][0])
    cy = max(c1[3][1],c2[3][1])
    newRect.append([cx,cy])
    return newRect

#given a list of corners that represent the corners of a box,
#find the center of that box
def findCenterCoor(c1):
    width = abs(c1[0][0]-c1[1][0])
    height = abs(c1[0][1]-c1[3][1])
    return([c1[0][0]+(width/2.0), c1[0][1]+(height/2.0)])

#take two points and find their slope
def findSlope(p1,p2):
    if(p1[0]-p2[0] == 0):
        return np.inf

    return (p1[1]-p2[1])/(p1[0]-p2[0])

#takes point and set of corners and checks if the point is within the bounds
def isInside(p1,c1):
    if(p1[0] >= c1[0][0] and p1[0] <= c1[1][0] and p1[1] >= c1[0][1] and p1[1] <= c1[2][1]):
        return True
    else:
        return False

def findArea(c1):
    return abs(c1[0][0]-c1[1][0])*abs(c1[0][1]-c1[3][1])






if __name__ == "__main__":
    bndingBx = []#holds bounding box of each countour
    corners = []

    img = cv2.imread('linear.png',0) #read image

    #perform gaussian blur (5*5)
    blur = cv2.GaussianBlur(img,(5,5),0)
    #apply adaptive threshold to image
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    th3 = cv2.bitwise_not(th3)
    #Otsu method if preferred
    # ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #reassign contours to the filled in image
    contours, heirar = cv2.findContours(th3, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #find the rectangle around each contour
    for num in range(0,len(contours)):
        #make sure contour is for letter and not cavity
        if(heirar[0][num][3] == -1):
            left = tuple(contours[num][contours[num][:,:,0].argmin()][0])
            right = tuple(contours[num][contours[num][:,:,0].argmax()][0])
            top = tuple(contours[num][contours[num][:,:,1].argmin()][0])
            bottom = tuple(contours[num][contours[num][:,:,1].argmax()][0])
            bndingBx.append([top,right,bottom,left])

    #find the edges of each bounding box
    for bx in bndingBx:
        corners.append(findCorners(bx))

    #draw the countours on thresholded image
    #x,y,w,h = cv2.boundingRect(th3)
    imgplot = plt.imshow(img,'gray')
    #draw the box
    # for bx in corners:
    #     plt.plot([bx[0][0],bx[1][0]],[bx[0][1],bx[1][1]],'g-',linewidth=2)
    #     plt.plot([bx[1][0],bx[2][0]],[bx[1][1],bx[2][1]],'g-',linewidth=2)
    #     plt.plot([bx[2][0],bx[3][0]],[bx[2][1],bx[3][1]],'g-',linewidth=2)
    #     plt.plot([bx[3][0],bx[0][0]],[bx[3][1],bx[0][1]],'g-',linewidth=2)

    ###############################################
    ####HERE WE WILL COMBINE DOTS WITH THEIR RESPECTIVE I'S A j'S
    #-IDEAS, USE IMAGE PROCESSING TO FIND LETTERS THAT HAVE
    #TOO SMALL OF AN AREA (LESS THAN AVG-STD) AND HAVE A CIRCULITY OF SOME RATING
    ##THOUGHT TWO,
    plt.clf()
    err = 2 #error value for minor/major axis ratio
    #list will hold the areas of each bounding boxes
    Area = []
    #go through each corner and append its area to the list
    for corner in corners:
        Area.append(findArea(corner))
    Area = np.asarray(Area) #organize list into array format
    avgArea = np.mean(Area) #find average area
    stdArea = np.std(Area) #find standard deviation of area
    outlier = (Area < avgArea - stdArea) #find the out liers, these are probably the dots
    for num in range(0,len(outlier)): #go through each outlier
        dot = False
        if(outlier[num]): #if the outlier is a dot, perform operations
            #create new image of black pixels
            black = np.zeros((len(img),len(img[0])),np.uint8)
            #add white pixels in the region that contains the outlier
            cv2.rectangle(black,(corners[num][0][0],corners[num][0][1]),(corners[num][2][0],corners[num][2][1]),(255,255),-1)
            #perform bitwise operation on original image to isolate outlier
            fin =  cv2.bitwise_and(th3,black)
            #find the contours of this outlier
            tempCnt,tempH = cv2.findContours(fin,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            #loop, due to structure of countours
            for cnt in tempCnt:
                #create bounding rectangle of contour
                rect = cv2.minAreaRect(cnt)
                #calculate major and minor axis
                axis1 = rect[1][0]/2.0
                axis2 = rect[1][1]/2.0
                if(axis1 != 0 and axis2 != 0): # do not perform if image has 0 dimension
                    ratio = axis1/axis2 #calculate ratio of axis
                    #if ratio is close to 1 (circular), then most likely a dot
                    if ratio > 1.0 - err and ratio < err + 1.0:
                        dot = True
            #if contour is a dot, we want to connect it to the closest
            #bounding box that is beneath it
            if dot:
                bestCorner = corners[num]
                closest = np.inf
                for crn in corners: # go through each set of corners
                    #find width and height of bounding box
                    width = abs(crn[0][0]-crn[1][0])
                    height = abs(crn[0][1]-crn[3][1])
                    #check to make sure character is below in position (greater y value)
                    if(corners[num][0][1] > crn[0][1]):
                        continue #if it's above the dot we don't care
                    elif dist(corners[num][0],crn[0]) < closest and crn != corners[num]:# and (findSlope(findCenterCoor(corners[num]),crn[0])) > 0:
                        # if(findArea(mergeBoxes(corners[num],crn))> avgArea+stdArea):
                        #     continue
                        #check the distance if it is below the dot
                        cent = findCenterCoor(crn)
                        bestCorner = crn
                        closest = dist(corners[num][0],crn[0])
                #modify the coordinates of the pic to include the dot
                #print(bestCorner)
                newCorners = mergeBoxes(corners[num],bestCorner)
                corners.append(newCorners)
                #print(newCorners)
                corners[num][0][0] = 0
                corners[num][0][1] = 0
                corners[num][1][0] = 0
                corners[num][1][1] = 0
                corners[num][2][0] = 0
                corners[num][2][1] = 0
                corners[num][3][0] = 0
                corners[num][3][1] = 0
                bestCorner[0][0] = 0
                bestCorner[0][1] = 0
                bestCorner[1][0] = 0
                bestCorner[1][1] = 0
                bestCorner[2][0] = 0
                bestCorner[2][1] = 0
                bestCorner[3][0] = 0
                bestCorner[3][1] = 0

    ###############################################
    # Take letters and turn them into objects
    AllLetters = []
    counter = 0
    for bx in corners:
        width = abs(bx[1][0] - bx[0][0])
        height = abs(bx[3][1] - bx[0][1])
        if width*height == 0:
            continue
        plt.plot([bx[0][0],bx[1][0]],[bx[0][1],bx[1][1]],'g-',linewidth=2)
        plt.plot([bx[1][0],bx[2][0]],[bx[1][1],bx[2][1]],'g-',linewidth=2)
        plt.plot([bx[2][0],bx[3][0]],[bx[2][1],bx[3][1]],'g-',linewidth=2)
        plt.plot([bx[3][0],bx[0][0]],[bx[3][1],bx[0][1]],'g-',linewidth=2)
        newLetter = Letter.Letter([bx[0][0],bx[0][1]],[height,width],counter)
        AllLetters.append(newLetter)
        counter+=1
    plt.imshow(th3,'gray')
    plt.show()
    plt.clf()
    # sort letters
    AllLetters.sort(key=lambda letter: letter.getY()+letter.getHeight())

    #project the y coordinates of the letters on to
    # the y axis
    prjYCoords = []
    for letter in AllLetters:
        prjYCoords.append(letter.getY()+letter.getHeight())
        plt.plot([letter.getX(),letter.getX()+letter.getWidth()],[letter.getY(),letter.getY()],'b-',linewidth=2)
        plt.plot([letter.getX()+letter.getWidth(),letter.getX()+letter.getWidth()],[letter.getY(),letter.getY()+letter.getHeight()],'b-',linewidth=2)
        plt.plot([letter.getX()+letter.getWidth(),letter.getX()],[letter.getY()+letter.getHeight(),letter.getY()+letter.getHeight()],'b-',linewidth=2)
        plt.plot([letter.getX(),letter.getX()],[letter.getY()+letter.getHeight(),letter.getY()],'b-',linewidth=2)


    for c in prjYCoords:
        plt.plot(0,c,'ro');

    #find distances between coordinates
    coorDists = [0]
    for num in range(1,len(prjYCoords)):
        valCur = prjYCoords[num]
        valPast = prjYCoords[num-1]
        coorDists.append(valCur-valPast)

    coorDists_c = []
    for num in range(0,len(coorDists)):
        if(coorDists[num] > 5):
            coorDists_c.append(coorDists[num])

    # #find division in distance data
    res,thval = findThresh(coorDists)
    #use division to distinguish between paragraphs and sentences
    lines = [[AllLetters[0]]]
    IDS = [[AllLetters[0].getID()]]
    count = 0
    #go through each letter
    # for num in range(1,len(AllLetters)):
    #     print(AllLetters[num].getArea())
    #     currPos = AllLetters[num].getY() + AllLetters[num].getHeight()
    #     prevPos = AllLetters[num-1].getY() + AllLetters[num-1].getHeight()
    #     #if the distance is below the threshold then letter is part of the same line
    #     if(currPos-prevPos<thval):
    #         lines[count].append(AllLetters[num])
    #         IDS[count].append(currPos)
    #     else:
    #         count +=1
    #         lines.append([AllLetters[num]])
    #         IDS.append([currPos])
    #
    # plt.clf()
    # for pos in IDS:
    #     linepos = max(pos)
    #     plt.plot([0,5000],[linepos,linepos],'r--')
    # #following list will find median points of division
    start = 0
    end = 0
    asd = 1.0
    meanCoord = float(sum(coorDists))/float(len(coorDists))
    stdCoord = np.std(coorDists)
    plt.clf()
    plt.plot(coorDists)
    plt.plot([0,400],[meanCoord+asd*stdCoord,meanCoord+asd*stdCoord],'r-')
    plt.show()

    medPoints = []
    for num in range(0,len(coorDists)):
        if coorDists[num] > meanCoord + asd*stdCoord and end == 0:
            start = num
        if coorDists[num] > meanCoord + asd*stdCoord and start > 0:
            end = num
            medPoints.append(int(start+(end-start)/2.0))
            start = num
    medPoints.append(start)

    medPoints.insert(0,0)
    #perform correction on list

    lines = []
    for num in range(0,len(medPoints)):
        lines.append(prjYCoords[medPoints[num]])
        print(medPoints[num])
        plt.plot([0,5000],[prjYCoords[medPoints[num]],prjYCoords[medPoints[num]]],'r-')
    imgplot = plt.imshow(img,'gray')
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()




    ######################not used functions
    # def meanShift1D(points):
    #     #find the minimum and maximum points
    #     minP = min(points)
    #     maxP = max(points)
    #     #number of points
    #     n = len(points)
    #     #bandwidth
    #     h = 0.1

    #function originally used to fill in cavities
    #faster alternative solution was chosen instead
    # def fillImage(img):
    #     h, w = img.shape[:2]
    #     mask = np.zeros((h+2,w+2),np.uint8)
    #     im_ff = img.copy()
    #
    #     #find the contours in the image
    #     contours, heirar = cv2.findContours(th3, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #
    #     for num in range(0,len(contours)):
    #         if(heirar[0][num][3] != -1):
    #             #find centroid of contour
    #             cnt = contours[num]
    #             #find the boundries of the contour
    #             left = tuple(cnt[cnt[:,:,0].argmin()][0])
    #             right = tuple(cnt[cnt[:,:,0].argmax()][0])
    #             top = tuple(cnt[cnt[:,:,1].argmin()][0])
    #             bottom = tuple(cnt[cnt[:,:,1].argmax()][0])
    #             #find centere coordinates of cavity --- we can do better
    #             cx = left[0] + (right[0] - left[0])/2
    #             cy = top[1] + (bottom[1] - top[1])/2
    #             #perform flood fill on the center of the contour
    #             cv2.floodFill(im_ff,mask,(cx,cy),255)
    #     return img | im_ff
