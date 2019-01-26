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

    # Take letters and turn them into objects
    AllLetters = []
    for bx in corners:
        width = abs(bx[1][0] - bx[0][0])
        height = abs(bx[3][1] - bx[0][1])
        newLetter = Letter.Letter([bx[0][0],bx[0][1]],[height,width])
        AllLetters.append(newLetter)
    plt.clf()
    #sort letters
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

    xvals = range(0,len(coorDists))
    # xvals = []
    # for v in range(0,len(coorDists)):
    #     xvals.append((1.0*v)/len(coorDists))
    #following list will find median points of division
    start = 0
    end = 0
    meanCoord = sum(coorDists)/len(coorDists)
    stdCoord = np.std(coorDists)
    medPoints = []
    for num in range(0,len(coorDists)):
        if coorDists[num] > meanCoord + stdCoord and end == 0:
            start = num
        if coorDists[num] > meanCoord + stdCoord and start > 0:
            end = num
            medPoints.append(start+(end-start)/2.0)
            start = num



    # plt.clf()
    # plt.plot(xvals, coorDists)
    # plt.plot(medPoints,[30]*len(medPoints),'ro')
    # plt.plot(xvals,[sum(coorDists)/len(coorDists)+np.std(coorDists)]*len(coorDists))
    # plt.show()
    plt.clf()
    #
    for num in range(0,len(medPoints)):
        plt.plot([0,1000],[medPoints[num],medPoints[num]],'r-')
    imgplot = plt.imshow(img,'gray')
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
