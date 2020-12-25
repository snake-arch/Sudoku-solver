import argparse
import cv2
import numpy as np
def reform(numbers):
    numbers = np.array(numbers)
    newarr = numbers.reshape(9, 9)
    newerarr = newarr.tolist()
    return newerarr
def valid(bo,num,pos):
    #checkrow
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1]!=i:
            return False
    #check col
    for i in range(len(bo[0])):
        if bo[i][pos[1]] == num and pos[0]!=i:
            return False
    #check small box
    box_X = pos[1]//3
    box_Y = pos[0]//3
    for i in range(box_Y*3,box_Y*3+3):
        for j in range(box_X*3,box_X*3+3):
            if bo[i][j]==num and (i,j)!=pos:
                return False
    return True

def print_board(bo):
    for i in range(len(bo)):
        if i%3==0 and i!=0:
            print("- - - - - - - - - - - - - -")

        for j in range(len(bo[0])):
            if j % 3 == 0 and j!=0:
                print(" | ",end="")

            if j==8:
                print(bo[i][j])

            else:
                print(str(bo[i][j])+" ",end="")


def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j]==0:
                return (i,j) #row , column
    return None

def solve(bo):
    find = find_empty(bo)
    if not find:
        return True
    else:
        row,col = find
    for i in range(1,10):
        if valid(bo,i,(row,col)):
            bo[row][col]=i
            if solve(bo):
                return True

            bo[row][col]=0

    return False
def displayImgWNos(img,numbers,color =(255,0,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range(0,9):
        for y in range(0,9):
            if numbers[(y*9)+x] != 0:
                cv2.putText(img,str(numbers[(y*9)+x]),(x*secW+int(secW/2)-18
                            ,int((y+0.8)*secH)),cv2.FONT_HERSHEY_COMPLEX
                            ,1.4,color,2,cv2.LINE_AA)
    return img
from matplotlib import pyplot as plt
#image= cv2.imread('C:/Users/maind/Desktop/cont1.png')
#print("width: {} pixels".format(img.shape[1]))
#print("height: {} pixels".format(img.shape[0]))
#print("channels: {} ".format(img.shape[2]))
#cv2.imshow('Image',img)
#b,g,r=img[249,250]
#corner = img[0:120,0:120]
#img[0:120,0:120]=(0,255,0)
#cv2.imshow("Updated",img)
#canvas = np.zeros((300,300,3),dtype="uint8")
#cv2.line(canvas,(0,0),(300,300),(0,128,128))
#cv2.line(canvas,(300,0),(0,300),(0,128,128))
#arr=[range(0,300,20),range(0,300,20)]
#for (row,x) in enumerate(arr[0]):
  #  for (col,y) in enumerate(arr[1]):
 #       cv2.rectangle(canvas, (x,y), (x+10,y+10), (255, 0, 0), -1)
#cv2.circle(canvas,(canvas.shape[1]//2,canvas.shape[0]//2),50,(255,255,255),-1)

#M = np.float32([[1, 0, 25], [0, 1, 50]])
#shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
#cv2.imshow("Shifted Down and Right", shifted)
#[1,0,t] t gives left or right - is left and vice versa
#[0,1,t] t gives up or down - is up and vice versa
#M = np.float32([[1, 0, -50], [0, 1, -90]])
#shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
#cv2.imshow("Shifted Up and Left", shifted)

#rotate

#x,y = image.shape[:2]
#center = (x//2,y//2)
#M = cv2.getRotationMatrix2D(center,45,0.5)# points to perfrom rotation as center
#then angle of rotation then magnification
#rotated = cv2.warpAffine(image,M,(x,y))
#image to warp according to warp matrix M and the height and length of the image
#cv2.imshow('rotated',rotated)

#resize
#small=cv2.resize(image,(150,150),interpolation=cv2.INTER_AREA)
#cv2.imshow('small',small)

#flipping

#for i in range(-1,2):
 #   flipped = cv2.flip(image, i)
  #  cv2.imshow('image', flipped)
   # cv2.waitKey(0)

#cropping
#cropped = image[0:150,0:150]
#cv2.imshow('cropped',cropped)


#blue=cv2.imshow("img",image[:,:,0])
#green=cv2.imshow("img1",image[:,:,1])
#red=cv2.imshow("img2",image[:,:,2])
#cv2.imshow('img3',image)
#also split func can be used
#(B,G,R)= cv2.split(image)
# then these can can pe merged
#merged = cv2.merge([B,G,R])
#zeros = np.zeros(image.shape[:2], dtype = "uint8")
#only redcv2.imshow("Red", cv2.merge([zeros, zeros, R]))
# only greencv2.imshow("Green", cv2.merge([zeros, G, zeros]))
#oly red cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))

#diff color packages
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Gray", gray)
#hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#cv2.imshow("HSV", hsv)
#lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#cv2.imshow("L*a*b*", lab)

#using masks with threshold value
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(image, (5, 5), 0)
#first convert to gray and blur it
#(T, thresh) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)
#cv2.imshow("Threshold Binary", thresh)
#threshold func first arg is img, value , max value, thresh_binary means it will convert gretater val tu 255
#(T, threshInv) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV)
#cv2.imshow("Threshold Binary Inverse", threshInv)
#for binary inv
#cv2.imshow("Coins", cv2.bitwise_and(image, image, mask =threshInv))
#cv2.waitKey(0)

#soduko solver
from keras.models import load_model
import pickle
# To stack all the resulting images final main function
def stackImages(imgArray,scale): #ye toh maa chudaye
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver
# 1 preprocessing the image
def preProcess(img):
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)
    imgx=cv2.adaptiveThreshold(imgBlur,255,1,1,11,2)
    return imgx
    # img to be applied to thresh, max value ie thresh val,type of adaptive method mean or gaussian
    # ,threshold type binary or inverse,neighbour pixels to be used nxn,value to subtracted from the mean
# 3rd step finding the biggest contour
def biggestContour(contours):
    biggest=np.array([])
    for i in contours:
        area=cv2.contourArea(i)
        max=0
        if area>50:
            peri=cv2.arcLength(i,True)#gives the perimeter
            approx=cv2.approxPolyDP(i,0.02*peri,True)#gives the coordinates of the corresponding contour
            if area>max and len(approx)==4:
                biggest=approx
                max=area
    return biggest,max
# 3rd step reordering points
def reorder(points):
    points=points.reshape((4,2))
    newpoints=np.zeros((4,1,2),dtype=np.int32)
    add=points.sum(1)
    newpoints[0]=points[np.argmin(add)]
    newpoints[3]=points[np.argmax(add)]
    diff=np.diff(points,axis=1)
    newpoints[1]=points[np.argmin(diff)]
    newpoints[2]=points[np.argmax(diff)]
    return newpoints
# 4 th step we split the image into 81 boxes since images can also be represented as numpy arrays
def splitBoxes(img):
    rows=np.vsplit(img,9)
    boxes=[]
    for r in rows:
        cols=np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes
# 4th step split and send to classify
def initializePredictionModel():
    #model = load_model('C:/Users/maind/PycharmProjects/GauravP1/venv/model_trained.p')
    pickle_in = open('C:/Users/maind/PycharmProjects/GauravP1/venv/model_trained.p', 'rb')
    model = pickle.load(pickle_in)
    return model
# 4th step getting the label
def getPrediction(boxes,model):
    result=[]
    for image in boxes:
        #prepare the image
        img= np.asarray(image)
        img=img[4:img.shape[0]-4,4:img.shape[1]-4]
        img = cv2.resize(img,(32,32))
        img =img/255
        img = img.reshape(1,32,32,1)
        # get prediction
        classIndex = int(model.predict_classes(img))

        predictions = model.predict(img)
        #print(predictions)
        probVal = np.amax(predictions)
        #print(classIndex, probVal)
        if probVal>0.8:
            result.append(classIndex)
        else:
            result.append(0)
    return result

def displayImgWNos(img,numbers,color =(255,0,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range(0,9):
        for y in range(0,9):
            if numbers[(y*9)+x] != 0:
                cv2.putText(img,str(numbers[(y*9)+x]),(x*secW+int(secW/2)-18
                            ,int((y+0.8)*secH)),cv2.FONT_HERSHEY_COMPLEX
                            ,1.4,color,2,cv2.LINE_AA)
    return img

model= initializePredictionModel()
# 1 prepare the image
imageX= cv2.imread('C:/Users/maind/Desktop/sudoku1.webp')
#imageX= cv2.imread('C:/Users/maind/Desktop/sudoku3.png')
#cv2.imshow("x",image)
image=cv2.resize(imageX,(450,450))
imgBlank=np.zeros((450,450,3),np.uint8)
imgThresh=preProcess(image)

 # 2 find contours
imgContours=image.copy() #this copy will contain all contours
imgBigContour=image.copy() #this coppy will contain the biggest contours
conts,heirarchy=cv2.findContours(imgThresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours,conts,-1,(0,255,0),3)#draw contours on copy of real image

# 3 find the biggest ie imgBigContour
width=450
height=450
biggest,max_area=biggestContour(conts) #biggest stores coordinates of biggest contour
if biggest.size!=0:
    biggest=reorder(biggest)#reorder the points for consistency
    cv2.drawContours(imgBigContour,biggest,-1,(0,255,0),25)#draw the biggest
    pts1=np.float32(biggest)#the points that need to be warped for ouur need
    pts2=np.float32([[0,0],[450,0],[0,450],[450,450]])#points are to be warped around these points ie the chars of the actucal image
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored=cv2.warpPerspective(image,matrix,(width,height))
    imgDetected=imgBlank.copy()
    imgWarpColored=cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

# 4 split and send to classify
imgSolvedDigits=imgBlank.copy()
boxes= splitBoxes(imgWarpColored)
numbers=getPrediction(boxes,model)
#print(numbers)

imgBlank= np.zeros((450,450,3),np.uint8)
imgDetectedDigits=imgBlank.copy()
imgDetectedDigits=displayImgWNos(imgDetectedDigits,numbers,color=(255,0,255))
numbers1 = reform(numbers)
solve(numbers1)
x=np.array(numbers1).reshape(81,)
Alldigits=displayImgWNos(imgBlank.copy(),x,color=(255,0,255))
imgWarpColored1=imgWarpColored.copy()
imgSolved=displayImgWNos(imgWarpColored1,x,color=(0,0,255))

#numbers=np.asarray(numbers)
#posArray=np.where(numbers> 0,0,1)
#print(numbers)



#display result
imageArray=([image,imgThresh,imgContours,imgWarpColored],[imgDetectedDigits,Alldigits,imgSolved,image])
stacked=stackImages(imageArray,0.5)
cv2.imshow('Solving process',stacked)


cv2.waitKey(0)




























