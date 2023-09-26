import cv2
import sys
import numpy as np

# recebe img de entrada
# int m de matiz, 0 a 360
# int x de faixa, onde se substitui m-x ate m+x



# 1920x1080 with some room to no crop at the top and bottom of the screen and proper aspect ratio
WIDTH_CAP = 1635
HEIGHT_CAP = 920


def try_resize_to_fit_screen(img):
    if img.shape[0] > WIDTH_CAP or img.shape[1] > HEIGHT_CAP:
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
    # convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # split channels
    h, s, v = cv2.split(hsv)

    # TODO: if m+x > 360, it will no work with hue 0, due to it being a circle
    # wont work with m =~ 0 or m =~ 360

    # find hue in range m-x to m+x
    #hlow = m - x if m - x >= 0 else m-x + 360
    #hhigh = m + x if m + x <= 360 else m+x - 360
    #print('hlow: {}, hhigh: {}'.format(hlow, hhigh))
    #hmask = cv2.inRange(h, hlow, hhigh)
    print('hlow: {}, hhigh: {}'.format(m-x, m+x))
    hmask = cv2.inRange(h, m-x, m+x)

    h[hmask > 0] = (h[hmask > 0] + 180) % 360


    newimg = cv2.merge((h, s, v))
    rgbimg = cv2.cvtColor(newimg, cv2.COLOR_HSV2RGB)

 
    return rgbimg

def show_comparison(img, img2):
    # concat both images side by side
    big_img = cv2.hconcat([img, img2])
    # resize img to fit screen.
    # the image can go out of the screen bounds if it's too big
    # cv opens image 1 to 1 in pixel ratio
    big_img = try_resize_to_fit_screen(big_img)

    cv2.imshow('Image', big_img)
    # checks if any key was pressed or the 'X' button in the window was pressed
    while cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) > 0:
        if cv2.waitKey(100) > 0:
            break
    cv2.destroyAllWindows()

def main():
    img_path = sys.argv[1]
    m = int(sys.argv[2])
    x = int(sys.argv[3])
    img = cv2.imread(img_path)

    #img to rgb 
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        print('Could not open or find the image:', img_path)
        exit(0)

    # normalize img to 0-1
    img = img.astype(np.float32)/255
    img2 = invert_hue(img, m, x)

    show_comparison(img, img2)


main()