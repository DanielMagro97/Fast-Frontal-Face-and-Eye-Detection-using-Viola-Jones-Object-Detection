import cv2
import os


# Function which normalises all the negative images to 250x250 grayscale
def normalise_neg_images():
    # create a directory for the normalised negative images if one doesn't already exist
    if not os.path.exists('neg'):
        os.makedirs('neg')

    img_num = 1
    for dir in ['neg_homeobjects', 'neg_random', 'neg_dogs', 'neg_flowers', 'neg_food', 'neg_cars', 'neg_natural']:
        for img_path in os.listdir("neg_raw/"+dir):
            try:
                # load the image from disk in grayscale
                img = cv2.imread("neg_raw/"+str(dir)+'/'+str(img_path), cv2.IMREAD_GRAYSCALE)
                # resize the grayscale image to 250x250
                resized_img = cv2.resize(img, (250,250))
                # rewrite the 250x250 grasycale image to disk
                cv2.imwrite("neg/"+str(img_num)+".jpg", resized_img)
                # increment the img_num so the next image gets a unique id
                img_num += 1
            except Exception as e:
                print("neg_raw/"+str(dir)+'/'+str(img_path))
                print(str(e))


# Function which creates the bg.txt file for all negatives.
# If running on Windows, change newline symbol to Linux for OpenCV functions
def create_bg_txt():
    for img in os.listdir('neg'):
        line = 'neg/' + img + '\n'
        with open('bg.txt', 'a') as f:
            f.write(line)


def create_info_txt():
    for img in os.listdir('pos'):
        line = 'pos/' + img + ' 1 0 0 178 218\n'
        with open('info.txt', 'a') as f:
            f.write(line)


normalise_neg_images()
create_bg_txt()
create_info_txt()
