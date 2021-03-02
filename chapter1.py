import cv2
import numpy
import time
print("Package imported")

# to read pictures
# img=cv2.imread("Resources/lena.jpg")
# cv2.imshow("Output",img)
# cv2.waitKey(0)

# to read videos
# cap=cv2.VideoCapture("Resources/videoplayback.mp4")
# while True:
#     success, img=cap.read()
#     cv2.imshow("Video",img)
#     if cv2.waitKey(1) & 0xFF==ord('q'):
#         break

# Some useful functions converting RGB to GRay scale
# img=cv2.imread("Resources/lena.jpg")
# imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray image',imgGray)
# cv2.waitKey(0)

# Another method to display in garyscale
# img=cv2.imread("Resources/lena.jpg")
# cv2.imshow("Output",cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
# cv2.waitKey(0)

# Converting picture to blur
# img=cv2.imread("Resources/lena.jpg")
# imgBlur=cv2.GaussianBlur(img,(11,11),0)
# cv2.imshow("Blur picture",imgBlur)
# cv2.waitKey(0)

# Finding the edges in the picture and adding the dialation,erosion,gradient to the edges
# img=cv2.imread("Resources/Hassam.jpeg")
# kernel = numpy.ones((2,2),numpy.uint8)
# imgEdges=cv2.Canny(img,100,100)
# dilation = cv2.dilate(imgEdges,kernel,iterations = 1)
# erosion = cv2.erode(dilation,kernel,iterations = 1)
# gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)
# cv2.imshow("Edge picture",gradient)
# cv2.waitKey(0)

# Resizing the images and  cropping the image
# img=cv2.imread("Resources/fayez.jpeg")
# print(img.shape)
# imgResized=cv2.resize(img,(500,500))
# print(imgResized)
# # print(img.shape[0]) #height
# # print(img.shape[1]) #width
# # print(img.shape[2]) #channels
# imgCropped=img[0:600,300:500]
# cv2.imshow("Cropped Image",imgCropped)
# cv2.imshow("Original image",img)
# cv2.waitKey(0)

#Shape and lines
#zero represent black pixels
# img=numpy.zeros((500,500,3),numpy.uint8)
# # print(img.shape)
# # img[0:100,0:400]=255,0,0   #idher height and width de rha ho mai ke ausi area mai color show ho ausle ilawa zeros(black) he show ho
# # img[:]=0,255,0   #it means ke pore area mai co,or show krdo
# # cv2.line(img,(0,0),(250,400),(0,0,255),5)  #width,height
# cv2.line(img,(500,0),(0,img.shape[0]),(0,0,255),5)  #img.shape[0]=height 100%,5 is the thinkness
# cv2.circle(img,(249,249),15,(255,0,0),cv2.FILLED)
# cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(125,125,125),5)
# cv2.rectangle(img,(100,100),(400,400),(0,255,0),5)
# cv2.putText(img,"Hassam \"The warrior\"",(90,25),cv2.FONT_ITALIC,1,(255,255,255),3)
# cv2.imshow("Grid",img)
# cv2.waitKey(0)

# wrap prespective
#https://www.mobilefish.com/services/record_mouse_coordinates/record_mouse_coordinates.php
# img=cv2.imread("Resources/cards.jpg")
# pts1=numpy.float32([[149,425],[287,362],[324,483],[176,542]])
# pts2=numpy.float32([[0,0],[200,0],[0,200],[200,200]])
# M = cv2.getPerspectiveTransform(pts1,pts2)
# dst = cv2.warpPerspective(img,M,(200,200))
#
# # cv2.imshow("Cards",img)
# cv2.imshow("Cards",dst)
# cv2.waitKey(0)

#horizontal and vertical stack
#we cannot output different channels or different changed colors with python pictures in horizontal or vertical so we need a function youtube murtzaa
# img=cv2.imread("Resources/lena.jpg")
# verticalImg=numpy.vstack([img,img])
# horizontalImg=numpy.hstack([img,img])
# cv2.imshow("Horizontal",horizontalImg)
# cv2.imshow("Vertical",verticalImg)
# cv2.waitKey(0)

#color detection
# def empty(a):
#     pass
# cv2.namedWindow("TrackBarss")
# cv2.resizeWindow("TrackBarss",640,240)
# cv2.createTrackbar("Hue Min","TrackBarss",0,179,empty)
# cv2.createTrackbar("Hue Max","TrackBarss",179,179,empty)
# cv2.createTrackbar("Sat Min","TrackBarss",0,255,empty)
# cv2.createTrackbar("Sat Max","TrackBarss",255,255,empty)
# cv2.createTrackbar("Value Min","TrackBarss",0,255,empty)
# cv2.createTrackbar("Value Max","TrackBarss",255,255,empty)
# while True:
#     img = cv2.imread("Resources/raibow.png")
#     imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h_min=cv2.getTrackbarPos("Hue Min","TrackBarss")
#     h_max=cv2.getTrackbarPos("Hue Max","TrackBarss")
#     s_min=cv2.getTrackbarPos("Sat Min","TrackBarss")
#     s_max=cv2.getTrackbarPos("Sat Max","TrackBarss")
#     v_min=cv2.getTrackbarPos("Value Min","TrackBarss")
#     v_max=cv2.getTrackbarPos("Value Max","TrackBarss")
#     print(h_min,h_max,s_min,s_max,v_min,v_max)
#     lower=numpy.array([h_min,s_min,v_min])
#     upper=numpy.array([h_max,s_max,v_max])
#     mask=cv2.inRange(imgHSV,lower,upper)
#     imgResult=cv2.bitwise_and(img,img,mask=mask)
#
#     # cv2.imshow("Rainbow",img)
#     # cv2.imshow("HSV",imgHSV)
#     cv2.imshow("Mask",mask)
#     cv2.imshow("Result", imgResult)
#     cv2.waitKey(1)

#shapes
# https://evergreenllc2020.medium.com/fundamentals-of-image-contours-3598a9bcc595Y
# def stackImages(scale,imgArray):
#     rows = len(imgArray)
#     cols = len(imgArray[0])
#     rowsAvailable = isinstance(imgArray[0], list)
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range ( 0, rows):
#             for y in range(0, cols):
#                 if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
#                 else:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
#         imageBlank = numpy.zeros((height, width, 3), numpy.uint8)
#         hor = [imageBlank]*rows
#         hor_con = [imageBlank]*rows
#         for x in range(0, rows):
#             hor[x] = numpy.hstack(imgArray[x])
#         ver = numpy.vstack(hor)
#     else:
#         for x in range(0, rows):
#             if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
#                 imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
#             else:
#                 imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
#             if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
#         hor= numpy.hstack(imgArray)
#         ver = hor
#     return ver
# def getContours(img):
#     contours,hierarchy  = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#     print(len(contours))
#
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         # print(area)
#         if area>500:
#             cv2.drawContours(imgContour, cnt, -1, (0, 255, 0), 3)
#             peri = cv2.arcLength(cnt,True)  #it is used to find contour perimeter of closed shape  If the second argument is True then it considers the contour to be closed.
#             # print(peri)
#             approx = cv2.approxPolyDP(cnt,0.02*peri,True) #approox perimeterr
#             print(len(approx))
#             objCor = len(approx)
#
#
#             if objCor ==3: objectType ="Tri"
#             elif objCor == 4:
#                 aspRatio = w/float(h)
#                 if aspRatio >0.98 and aspRatio <1.05: objectType= "Square"
#                 else:objectType="Rectangle"
#             elif objCor>4: objectType= "Circles"
#             else:objectType="None"
#
#             x, y, w, h = cv2.boundingRect(approx)
#             cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
#             cv2.putText(imgContour,objectType,
#                         (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
#                         (0,0,0),1)
#
#
# path = "Resources/shapes1.png"
# img = cv2.imread(path)
# imgContour = img.copy()
#
# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# imgCanny = cv2.Canny(imgGray,50,50)
# imgBlur = cv2.GaussianBlur(imgCanny,(7,7),1)
#
# getContours(imgBlur)
#
# imgBlank = numpy.zeros_like(img)
# imgStack = stackImages(1,([imgContour]))
#
# cv2.imshow("Stack", imgStack)
# cv2.waitKey(0)


#face detection vscola
#for custom haar classifier https://www.researchgate.net/publication/338680205_Building_Custom_HAAR-Cascade_Classifier_for_face_Detection
# faceCascade=cv2.CascadeClassifier("haar.xml")
# img=cv2.imread("Resources/faces.jpg")
# imgSize=cv2.resize(img,(1150,900))
# imgGray=cv2.cvtColor(imgSize,cv2.COLOR_BGR2GRAY)
#
#
# face=faceCascade.detectMultiScale(imgGray,1.7,4) #1.5 is the zoom in on a window scale #4=So there are a lot of face detection because of resizing the sliding window and a lot of false positives too. So to eliminate false positives and get the proper face rectangle out of detections, neighborhood approach is applied. It is like if it is in neighborhood of other rectangles than it is ok, you can pass it further. So this number determines the how much neighborhood is required to pass it as a face rectangle. In the same image when it is 1 : https://stackoverflow.com/questions/22249579/opencv-detectmultiscale-minneighbors-parameter
# for x,y,w,h in face:
#     rectangle=cv2.rectangle(imgSize,(x,y),(x+w,y+h),(0,255,0),3)
#
# cv2.imshow("Hassam",imgSize)
# cv2.waitKey(0)

#real time face detection using the phone camera
#https://towardsdatascience.com/a-guide-to-face-detection-in-python-3eab0f6b9fc1
# face_Cascade=cv2.CascadeClassifier("haar.xml")
# video=cv2.VideoCapture(0)
# address='https://192.168.18.126:8080/video'
# video.open(address)
#
# while True:
#     check,frame=video.read()
#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     face=face_Cascade.detectMultiScale(gray,1.4,5)
#
#     for x,y,w,h in face:
#         img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
#     cv2.imshow("Video real time",frame)
#     key=cv2.waitKey(1)
#
#     if key==ord('q'):
#         break
# video.release()
# cv2.destroyAllWindows()


