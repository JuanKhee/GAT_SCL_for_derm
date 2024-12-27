def normalise_pixel_value(img):
    img = img/255

    return img


def preprocess_images(images):
    images_processed = []
    for img in images:
        print(img)
        img_processed = normalise_pixel_value(img)
        images_processed.append(img_processed)

    return images_processed