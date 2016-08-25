import cv2
import os
from os import listdir, mkdir
from os.path import isfile, join, exists

def makeEmpty(path):

    if not exists(path):
        mkdir(path)
        return

    _files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    for _file in _files:
        os.remove(_file)

def doMasking(imageDirectory, maskedDirectory):
    makeEmpty(maskedDirectory)

    _images = [f for f in listdir(imageDirectory) if isfile(join(imageDirectory, f))]

    for _image in _images:
        image  = cv2.imread(join(imageDirectory, _image))
        result_image = image.copy()

        # Specify the trained cascade classifier
        face_cascade_name = './haarcascade/haarcascade_frontalface_alt.xml'

        # Create a cascade classifier
        face_cascade = cv2.CascadeClassifier()

        # Load the specified classifier
        face_cascade.load(face_cascade_name)

        # Preprocess the image
        grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayimg = cv2.equalizeHist(grayimg)

        # Run the classifiers
        faces = face_cascade.detectMultiScale(grayimg, 1.1, 2, 0 | cv2.CASCADE_SCALE_IMAGE, (30, 30))

        # print "Faces detected"

        if len(faces) != 0:  # If there are faces in the images
            for f in faces:  # For each face in the image

                # Get the origin co-ordinates and the length and width till where the face extends
                x, y, w, h = [v for v in f]

                # get the rectangle img around all the faces
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 5)
                sub_face = image[y:y + h, x:x + w]
                # apply a gaussian blur on this new recangle image
                sub_face = cv2.GaussianBlur(sub_face, (23, 23), 30)
                # merge this blurry rectangle to our final image
                result_image[y:y + sub_face.shape[0], x:x + sub_face.shape[1]] = sub_face
                # face_file_name = "./face_" + str(y) + ".jpg"
                # cv2.imwrite(face_file_name, sub_face)

        # cv2.imshow("Detected face", result_image)
        # print join(outPath, _image)
        cv2.imwrite(join(maskedDirectory, _image), result_image)