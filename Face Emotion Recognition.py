from deepface import DeepFace
import cv2
import matplotlib.pyplot




print("Do you want to predict emotions from Webcam or the Photo you have provided?")
print("Press 1 for Photo, 2 for Webcam")
a = input()

if a == '1':

    print("The image you have provided in the folder will be predicted now")
    pic = cv2.imread('image.png')

    # matplotlib.pyplot.imshow(pic)
    # matplotlib.pyplot.imshow(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
    # matplotlib.pyplot.show()

    predict = DeepFace.analyze(pic)

    type(predict)

    predict['dominant_emotion']

    cascade1 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    pic_grey = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

    face = cascade1.detectMultiScale(pic_grey,1.1,4)

    for(x1, y1, w1, h1) in face:
         cv2.rectangle(pic, (x1,y1), (x1+w1, y1+h1), (255,0,0), 2)
    #     matplotlib.pyplot.imshow(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
    #     matplotlib.pyplot.show()

    write = cv2.FONT_HERSHEY_PLAIN

    cv2.putText(pic,
                    predict['dominant_emotion'],
                    (25, 100),
                    write, 10,
                    (0, 0, 255),
                    2,
                    cv2.LINE_4) ;

    matplotlib.pyplot.imshow(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
    matplotlib.pyplot.show()


elif a == '2':
    print("You have chosen Webcam, It will be opened shortly")


    cascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise IOError("Cannot open Webcam")

    while True:
        ret, frames = cam.read()
        output = DeepFace.analyze(img_path=frames, actions=['emotion'], enforce_detection=False)

        grey2 = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

        face2 = cascade2.detectMultiScale(grey2, 1.1, 4)

        for (x2, y2, w2, h2) in face2:
            cv2.rectangle(frames, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 3)

        emotion = output["dominant_emotion"]

        text = str(emotion)

        cv2.putText(frames, text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 3)
        cv2.imshow('webcam', frames)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



