import cv2
import os, random   # for choosing a random image from a directory of sample images

# load pre-trained cascade classifiers
haar_face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
haar_eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

while True:
    # Ask the user whether they want to detect faces and eyes in their image, a sample image, or the webcam.
    user_choice = input("Kindly either enter the full path of an image you would like to detect faces and eyes in,\n"
                        + "or write 'img' to see the program work on an image from the evaluation set,\n"
                        + "or write 'cam' to see the program work on a feed from the webcam,\n"
                        + "or write 'quit' to quit.\n")

    if user_choice == "img":
        sample_imgs_path = "evaluation"
        random_img_path = sample_imgs_path + "\\" + random.choice(os.listdir("evaluation"))
        print(random_img_path)
        img = cv2.imread(random_img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect the faces in the image
        faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

        # print the number of faces found
        print('Faces found: ', len(faces))

        # go over list of faces and draw them as rectangles on original colored
        for (x, y, w, h) in faces:
            # draw a rectangle around the face, colour is BGR
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # set the region of interest of the image (for eye dection) to the area in which the face was detected
            img_roi = img[y:y + h, x:x + w]
            gray_roi = gray_img[y:y + h, x:x + w]

            # detect eyes using the eye cascade in the region of interest,
            # i.e. only in the area of the image where the face was detected
            eyes = haar_eye_cascade.detectMultiScale(gray_roi)  # , scaleFactor=1.1, minNeighbors=1)
            print('Eyes found: ', len(eyes))
            # go over the list of eyes and draw red rectangles around them on the coloured image
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img_roi, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
        cv2.imshow('Sample Image', img)
        print("Press Any Key to continue")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif user_choice == "cam":
        print("Press 'Esc' to stop")
        webcam_stream = cv2.VideoCapture(0)
        while 1:
            ret, cam_img = webcam_stream.read()
            gray_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2GRAY)
            faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(cam_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # set the region of interest of the image (for eye dection) to the area in which the face was detected
                img_roi = cam_img[y:y + h, x:x + w]
                gray_roi = gray_img[y:y + h, x:x + w]

                # detect eyes using the eye cascade in the region of interest,
                # i.e. only in the area of the image where the face was detected
                eyes = haar_eye_cascade.detectMultiScale(gray_roi)  # , scaleFactor=1.1, minNeighbors=1)
                # print('Eyes found: ', len(eyes))
                # go over the list of eyes and draw red rectangles around them on the coloured image
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(img_roi, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

            cv2.imshow('Webcam', cam_img)

            # find faces and eyes from webcam until Esc is pressed
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        cv2.destroyAllWindows()
        webcam_stream.release()

    elif user_choice == "quit":
        break

    else:
        # load image from disk
        img = cv2.imread(user_choice)
        # convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect the faces in the image
        faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

        # print the number of faces found
        print('Faces found: ', len(faces))

        # go over list of faces and draw green rectangles around them on the coloured image
        for (x, y, w, h) in faces:
            # draw a rectangle around the face, colour is BGR
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # set the region of interest of the image (for eye dection) to the area in which the face was detected
            img_roi = img[y:y+h, x:x+w]
            gray_roi = gray_img[y:y+h, x:x+w]

            # detect eyes using the eye cascade in the region of interest,
            # i.e. only in the area of the image where the face was detected
            eyes = haar_eye_cascade.detectMultiScale(gray_roi)  # , scaleFactor=1.1, minNeighbors=1)
            print('Eyes found: ', len(eyes))
            # go over the list of eyes and draw red rectangles around them on the coloured image
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img_roi, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

        # show image with rectangles
        cv2.imshow('Image', img)
        print("Press Any Key to continue")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
