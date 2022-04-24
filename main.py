import cv2
import numpy as np
import os

# Reduces te image by K size, ex: 16 = 4 bits
# and returns an image
def reduce_img(original_image ,K):
    Z = original_image.reshape((-1, 1))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    img = res.reshape((original_image.shape))

    return img

# Returns the number that is closest to K in a list
def closest(K, lst):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]

Ascii = { 0:"@", 30:"#", 50:"B", 100:"X", 150:"=", 200:"+", 230:":", 255:"," }


# Prints an image to the terminal according to the ascii dictionary
def convertAscii(image):
    line = ""

    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            k = image[i, j]
            key = closest(k, list(Ascii.keys()))
            line += Ascii.get(key)
        print(line)
        line = ""
    os.system('cls' if os.name == 'nt' else 'clear')


# Create an image.txt file with the current frame
def screenshot(image):
    ascii_image = []
    line = ""

    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            k = image[i, j]

            # Gets the closest number to the current value of k from the list of ascii keys
            line += Ascii.get(closest(k, list(Ascii.keys())))

        ascii_image.append(line)
        line = ""

    with open('image.txt', 'w') as f:
        for l in ascii_image:
            f.write(l + '\n')

if __name__ == '__main__':
    video = cv2.VideoCapture(0)

    while(True):
        ret, frame = video.read(0)

        # Scale down the image so the program does not slow down because of a high resolution
        scale_percent = 40
        height = int(frame.shape[0] * scale_percent / 100)
        width = int(frame.shape[1] * scale_percent / 100)
        dsize = (width, height)
        frame = cv2.resize(frame, dsize)

        # Turn the image grayscale and use k clustering to make it easier to convert to ascii
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = reduce_img(gray_frame, 64)
        convertAscii(gray_frame)


        cv2.imshow('video', frame)
        # Inputs
        key = cv2.waitKey(1)
        # Pressing space makes an image.txt file with the ascii output
        if key%256 == 32:
            screenshot(gray_frame)
        # Pressing escape exits the program
        if key%256 == 27:
            break


    video.release()

    cv2.destroyAllWindows()
