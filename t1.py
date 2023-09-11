import cv2
import sys

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

def main():
    img_path = sys.argv[1]
    img = cv2.imread(img_path)

    if img is None:
        print('Could not open or find the image:', img_path)
        exit(0)



    img = try_resize_to_fit_screen(img)
    cv2.imshow('Image', img)
    # checks if any key was pressed or the 'X' button in the window was pressed
    while cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) > 0:
        if cv2.waitKey(100) > 0:
            break
    cv2.destroyAllWindows()

main()