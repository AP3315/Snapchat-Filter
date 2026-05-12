import cv2
cap=cv2.VideoCapture(0)

eyescas = cv2.CascadeClassifier('frontalEyes35x16.xml')
nosecascade= cv2.CascadeClassifier('Nose18x15.xml')
facecas=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
gla=cv2.imread('glasses.png',-1)
mus=cv2.imread('mustache.png',-1)

while True:
    ret,frame=cap.read()

    if(ret==False):
        continue
    
    # face=facecas.detectMultiScale(frame,1.9,6) 
    # print(face)

    eyes=eyescas.detectMultiScale(frame,1.9,6)
    # print(eyes)
    
    # for i in eyes:
    #     cv2.rectangle(frame,(i[0],i[1]),(i[0]+i[3],i[1]+i[2]),(255,255,255),1)


    nose=nosecascade.detectMultiScale(frame,1.3,5)
    # print(nose)   

    # now simply overlay the glasses over the frame captured by webcam, we have starting points (x,y) as (eyes[0][0], eyes[0][1])
    # nw both gla and frame has images and hence (r,c, ch) i.e. (y, x, ch) for both 
    # if(len(eyes)!=0):
    #     gla=cv2.resize(gla,(eyes[0][3],eyes[0][2]))
    #     # print(gla.shape)
    #     for i in range(gla.shape[0]):
    #         for j in range(gla.shape[1]):
    #             if(gla[i,j,3]>0):
    #                 for k in range(gla.shape[2]-1):             
    #                     frame[eyes[0][1]+i,eyes[0][0]+j,k]=gla[i,j,k]

    # if(len(nose)!=0):
    #     mus=cv2.resize(mus,(nose[0][3]-15,nose[0][2]-15))    #also making mus size smaller , it big
    #     for i in range(mus.shape[0]):
    #         for j in range(mus.shape[1]):
    #             if(mus[i,j,3]>0):
    #                 for k in range(mus.shape[2]-1):
    #                     frame[28+nose[0][1]+i,9+nose[0][0]+j,k]=mus[i,j,k]    #also shifting nose to correct position


    if(len(eyes)) != 0:
        # Resize glasses ( w,h are swapped due to model issues) to match face's eye size
        gla_resized = cv2.resize(gla, (eyes[0][3], eyes[0][2])) 
        
        # Get position
        x, y = eyes[0][0], eyes[0][1]   #starting position
        h, w = gla_resized.shape[:2]    
        
        # Create mask from alpha channel
        # mask will be 2d array boolean array of exactly have same shape as gla_resized.shape
        # having true where gla_resized pixel have value in 4th channel(alpha channel) not zero and false otherwise
        mask = gla_resized[:, :, 3] != 0   
        
        # One line instead of triple loop!
        # frame[y:y+h, x:x+w] this selects the eye region from the face having exactly same shape as gla_resized
        # frame[y:y+h, x:x+w][mask] now this make array of array having shape (n,3) ie array having n elements and each element having 3 value(pixel)
        # this is nothing set all pixels from face where mask is true ie the glasses part, n pixels have true value in mask
        # mask is made from gla_resized
        # similarly gla_resized[:, :, :3] it is gla_resized with its 4th channel(alpha channle) dropped, since frame has also 3 channls only
        # gla_resized[:, :, :3][mask] and this array conating n elements and each n elements has 3 values (pixel value, bgr)
        # and n is the number of pixels where aplha is non zero
        # now both lhs and rhs have exactly same shape(n,3) and finnaly values from rhs will be copied to lhs andd lhs(frame), lhs will be updated
        # same goes with the nose too
        frame[y:y+h, x:x+w][mask] = gla_resized[:, :, :3][mask]

    if(len(nose)) != 0:
        # Resize mustache
        mus_resized = cv2.resize(mus, (nose[0][3]-10, nose[0][2]-10))   #also making mus size smaller , it big
        
        # Get position (with your offsets)
        x, y = nose[0][0]+8, nose[0][1]+28                                   #also shifting nose to correct position
        h, w = mus_resized.shape[:2]
        
        # Create mask from alpha channel
        mask = mus_resized[:, :, 3] != 0
        
        # One line instead of triple loop!
        frame[y:y+h, x:x+w][mask] = mus_resized[:, :, :3][mask]


    
    cv2.imshow('frame',frame)
    keypressed=cv2.waitKey(1)&0xFF
    if(keypressed==ord('q')):
        break

cv2.destroyAllWindows()
cap.release()
