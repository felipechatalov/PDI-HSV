import cv2
import sys
import numpy as np


# 1920x1080 with some room to no crop at the top and bottom of the screen and proper aspect ratio
WIDTH_CAP = 1635
HEIGHT_CAP = 920
def try_resize_to_fit_screen(img):
    if img.shape[0] > WIDTH_CAP or img.shape[1] > HEIGHT_CAP:
        # Get the width and height proportions and subtract is value each
        # iteration until it fits the screen properly
        h_proportion = img.shape[0] / img.shape[1]
        w_proportion = img.shape[1] / img.shape[0]

        h = img.shape[0]
        w = img.shape[1]

        while w > WIDTH_CAP:
            w -= 1
            h -= h_proportion
        
        while h > HEIGHT_CAP:
            h -= 1
            w -= w_proportion
        
        h = int(h)
        w = int(w)

        print('Resizing image to fit screen: {}x{} -> {}x{}'.format(img.shape[0], img.shape[1], h, w))

        img = cv2.resize(img, (w, h))
    return img

def invert_hue(img, m, x):
    # Convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split channels
    h, s, v = cv2.split(hsv)

    # Find hue in range m-x to m+x
    # hlow = m - x
    # hhigh = m + x
    hlow = m-x
    hhigh = m+x

    print('Inverting hue from {} to {}'.format(hlow, hhigh))
    # return a hue 'mask' where we going to invert it 
    hmask = cv2.inRange(h, hlow, hhigh)

    # If we see overlap on m-x and m+x, we separatly get the valuues
    # and use or operator on both masks, so we dont have any overlap.
    # Overlapping means that the hue is in both ranges, so it will be inverted 2x, so it will go back to the original colour
    if hlow < 0:
        hlow += 360
        hmasklow = cv2.inRange(h, hlow, 361)
        cv2.bitwise_or(hmask, hmasklow, hmask)
    elif hhigh > 360:
        hhigh -= 360
        hmaskhigh = cv2.inRange(h, -1, hhigh+1)
        cv2.bitwise_or(hmask, hmaskhigh, hmask)
    h[hmask > 0] = (h[hmask > 0] + 180) % 360

    # Merge channels back
    inv_img = cv2.merge((h, s, v))
    # Convert back to rgb
    img2 = cv2.cvtColor(inv_img, cv2.COLOR_HSV2BGR)
    return img2

def show_comparison(img, img2):
    # Concat both images side by side
    big_img = cv2.hconcat([img, img2])
    # Resize img to fit screen.
    # The image can go out of the screen bounds if it's too big,
    # cv opens image 1 to 1 in pixel ratio
    big_img = try_resize_to_fit_screen(big_img)

    cv2.imshow('Image', big_img)
    # Checks if any key was pressed or the 'X' button in the window was pressed
    while cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) > 0:
        if cv2.waitKey(100) > 0:
            break
    cv2.destroyAllWindows()

def main():
    # Read image, m and x values
    img_path = sys.argv[1]
    m = int(sys.argv[2])
    x = int(sys.argv[3])

    # Open image
    img = cv2.imread(img_path)

    # If not possible to open image
    if img is None:
        print('Could not open or find the image:', img_path)
        exit(0)

    # Normalize img to 0-1
    img = img.astype(np.float32)/255
    # Call function to invert the hue
    img2 = invert_hue(img, m, x)
    # Show the comparison on screen
    show_comparison(img, img2)

main()