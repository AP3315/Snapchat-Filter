import cv2
cap=cv2.VideoCapture(0)


nosecascade= cv2.CascadeClassifier('Nose18x15.xml')
facecas=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
mus=cv2.imread('mustache.png',-1)

while True:
    ret,frame=cap.read()

    if(ret==False):
        continue
    
    # face=facecas.detectMultiScale(frame,1.9,6) 
    # print(face)
    
    nose=nosecascade.detectMultiScale(frame,1.3,5)
    print(nose)
    
    # for i in nose:
    #     cv2.rectangle(frame,(i[0],i[1]),(i[0]+i[3],i[1]+i[2]),(255,255,255),1)

    if(len(nose)!=0):
        mus=cv2.resize(mus,(nose[0][3],nose[0][2]))
        for i in range(mus.shape[0]):
            for j in range(mus.shape[1]):
                for k in range(mus.shape[2]-1):
                    if(mus[i,j,3]>0):
                        frame[25+nose[0][1]+i,nose[0][0]+j,k]=mus[i,j,k]
    
    
    cv2.imshow('frame',frame)
    keypressed=cv2.waitKey(1)&0xFF
    if(keypressed==ord('q')):
        break

cv2.destroyAllWindows()
cap.release()
