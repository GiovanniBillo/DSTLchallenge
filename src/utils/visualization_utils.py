import matplotlib.pyplot as plt
import numpy as np

def M(image_id):    
    if DEBUG_MODE:
        print(f">> M called with: image_id={repr(image_id)}")

    # __author__ = amaia
    # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
    filename = os.path.join(RAW_DATA_DIR, 'sixteen_band', '{}_M.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img

def M3(image_id):    
    if DEBUG_MODE:
        print(f">> M called with: image_id={repr(image_id)}")

    filename = os.path.join(RAW_DATA_DIR, 'three_band', '{}.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img

def check_predict(id='6120_2_3'):    
    if DEBUG_MODE:
        print(f">> check_predict called with: id={repr(id)}")

    # model = get_unet()
    # model.load_weights('weights/unet_10_jk0.7878')

    # msk = predict_id(id, model, [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1])
    img = M(id)

    plt.figure()
    ax1 = plt.subplot(131)
    ax1.set_title('image ID:6120_2_3')
    ax1.imshow(img[:, :, 5], cmap=plt.get_cmap('gist_ncar'))
    # ax2 = plt.subplot(132)
    # ax2.set_title('predict bldg pixels')
    # ax2.imshow(msk[0], cmap=plt.get_cmap('gray'))
    # ax3 = plt.subplot(133)
    # ax3.set_title('predict bldg polygones')
    # ax3.imshow(mask_for_polygons(mask_to_polygons(msk[0], epsilon=1), img.shape[:2]), cmap=plt.get_cmap('gray'))

    plt.show()

