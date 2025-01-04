import cv2


def show_image(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return None


def print_image_detail(img):
    print('max and min pixel value')
    print(img.max(), img.min())

    print('image numpy size')
    print(img.shape)

    return None


if __name__ == "__main__":
    img = cv2.imread(
        r'C:\Users\HP-VICTUS\PycharmProjects\pythonProject\GAT_SCL_for_derm\dev_images\train\AK\ISIC_0025368.jpg')
    print_image_detail(img)
    show_image(img)