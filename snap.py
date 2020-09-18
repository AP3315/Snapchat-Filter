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

    if(len(eyes)!=0):
        gla=cv2.resize(gla,(eyes[0][3],eyes[0][2]))
        # print(gla.shape)
        for i in range(gla.shape[0]):
            for j in range(gla.shape[1]):
                for k in range(gla.shape[2]-1):             
                    if(gla[i,j,3]>0):
                        frame[eyes[0][1]+i,eyes[0][0]+j,k]=gla[i,j,k]


    nose=nosecascade.detectMultiScale(frame,1.3,5)
    # print(nose)   

    if(len(nose)!=0):
        mus=cv2.resize(mus,(nose[0][3]-15,nose[0][2]-15))
        for i in range(mus.shape[0]):
            for j in range(mus.shape[1]):
                for k in range(mus.shape[2]-1):
                    if(mus[i,j,3]>0):
                        frame[28+nose[0][1]+i,9+nose[0][0]+j,k]=mus[i,j,k] 
    
    cv2.imshow('frame',frame)
    keypressed=cv2.waitKey(1)&0xFF
    if(keypressed==ord('q')):
        break

cv2.destroyAllWindows()
cap.release()
