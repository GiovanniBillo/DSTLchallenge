import sys
DEBUG_MODE = '--debug' in sys.argv
# TODO: add more cli arguments that determine the amount of images in train and test(what now is s in the stick_all_train function) 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from shapely.wkt import loads as wkt_loads
import tifffile as tiff
import os
import random
# from keras.models import Model
# from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras import backend as K
# from sklearn.metrics import jaccard_similarity_score
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
from collections import defaultdict

from src.config import DATA_DIR, N_CLS, RAW_DATA_DIR, DF, GS, SB, ISZ, smooth

def _convert_coordinates_to_raster(coords, img_size, xymax):    
    if DEBUG_MODE:
        print(f">> _convert_coordinates_to_raster called with: coords={repr(coords)}, img_size={repr(img_size)}, xymax={repr(xymax)}")
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):    
    if DEBUG_MODE:
        print(f">> _get_xmax_ymin called with: grid_sizes_panda={repr(grid_sizes_panda)}, imageId={repr(imageId)}")

    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
    return (xmax, ymin)


def _get_polygon_list(wkt_list_pandas, imageId, cType):    
    if DEBUG_MODE:
        print(f">> _get_polygon_list called with: wkt_list_pandas={repr(wkt_list_pandas)}, imageId={repr(imageId)}, cType={repr(cType)}")

    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList

## THIS VERSION IS OLD AND NOT WORKING

# def _get_and_convert_contours(polygonList, raster_img_size, xymax):    
#     print(f">> _get_and_convert_contours called with: polygonList={repr(polygonList)}, raster_img_size={repr(raster_img_size)}, xymax={repr(xymax)}")

#     # __author__ = visoft
#     # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
#     perim_list = []
#     interior_list = []
#     if polygonList is None or polygonList.is_empty:
#         return None
#     for k in range(len(polygonList.geoms)):
#         poly = polygonList[k]
#         perim = np.array(list(poly.exterior.coords))
#         perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
#         perim_list.append(perim_c)
#         for pi in poly.interiors:
#             interior = np.array(list(pi.coords))
#             interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
#             interior_list.append(interior_c)
#     return perim_list, interior_list

def _get_and_convert_contours(polygonList, raster_img_size, xymax):    
    if DEBUG_MODE:
        print(f">> _get_and_convert_contours called with: polygonList={repr(polygonList)}, raster_img_size={repr(raster_img_size)}, xymax={repr(xymax)}")

    perim_list = []
    interior_list = []

    if polygonList is None or polygonList.is_empty:
        return None

    # Handle MultiPolygon or single Polygon
    polygons = polygonList.geoms if isinstance(polygonList, MultiPolygon) else [polygonList]

    for poly in polygons:
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)

        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)

    return perim_list, interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value=1):    
    if DEBUG_MODE:
        print(f">> _plot_mask_from_contours called with: raster_img_size={repr(raster_img_size)}, contours={repr(contours)}, class_value={repr(class_value)}")

    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    img_mask = np.zeros(raster_img_size, np.uint8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda=GS, wkt_list_pandas=DF):    
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
    if DEBUG_MODE:
        print(f">> generate_mask_for_image_and_class called with: raster_size={repr(raster_size)}, imageId={repr(imageId)}, class_type={repr(class_type)}, grid_sizes_panda={repr(grid_sizes_panda)}, wkt_list_pandas={repr(wkt_list_pandas)}")
        print("POLYGON LIST IS OF TYPE:", type(polygon_list))
        print("POLYGON LIST:", polygon_list)
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)
    return mask


def M(image_id):    
    if DEBUG_MODE:
        print(f">> M called with: image_id={repr(image_id)}")

    # __author__ = amaia
    # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
    filename = os.path.join(RAW_DATA_DIR, 'sixteen_band', '{}_M.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img

# TODO: images appear weird because of 8 spectral channels
# how to make them appear normal?
def M3(image_id):    
    if DEBUG_MODE:
        print(f">> M called with: image_id={repr(image_id)}")

    filename = os.path.join(RAW_DATA_DIR, 'three_band', '{}.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img

def stretch_n(bands, lower_percent=5, higher_percent=95):    
    if DEBUG_MODE:
        print(f">> stretch_n called with: bands={repr(bands)}, lower_percent={repr(lower_percent)}, higher_percent={repr(higher_percent)}")

    out = np.zeros_like(bands)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t

    return out.astype(np.float32)





def stick_all_train():    
    if DEBUG_MODE:
        print(f">> stick_all_train called with: ")


    # s = 835
    s = 400

    x = np.zeros((5 * s, 5 * s, 8))
    y = np.zeros((5 * s, 5 * s, N_CLS))

    ids = sorted(DF.ImageId.unique())
    print (len(ids))
    for i in range(5):
        for j in range(5):
            id = ids[5 * i + j]
            print(id)
            img = M(id)
            img = stretch_n(img)
            print (img.shape, id, np.amax(img), np.amin(img))
            x[s * i:s * i + s, s * j:s * j + s, :] = img[:s, :s, :]
            for z in range(N_CLS):
                y[s * i:s * i + s, s * j:s * j + s, z] = generate_mask_for_image_and_class(
                    (img.shape[0], img.shape[1]), id, z + 1)[:s, :s]

    print (np.amax(y), np.amin(y))
    
    np.save(f"{DATA_DIR}/x_trn_%d" % N_CLS, x)
    np.save(f"{DATA_DIR}/y_trn_%d" % N_CLS, y)

def stick_single_train():
    if DEBUG_MODE:
        print(f">> stick_all_train called")

    s = 400

    # Create output directories
    image_out_dir = os.path.join(DATA_DIR, "processed/train/images")
    mask_out_dir = os.path.join(DATA_DIR, "processed/train/masks")
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    ids = sorted(DF.ImageId.unique())
    print(f"Total IDs: {len(ids)}")
    
    counter = 0
    for i in range(5):
        for j in range(5):
            id = ids[5 * i + j]
            print(id)
            img = M(id)
            img = stretch_n(img)
            print(img.shape, id, np.amax(img), np.amin(img))

            img_patch = img[:s, :s, :]
            mask_patch = np.zeros((s, s, N_CLS))

            for z in range(N_CLS):
                mask = generate_mask_for_image_and_class((img.shape[0], img.shape[1]), id, z + 1)
                mask_patch[:, :, z] = mask[:s, :s]

            # Save image and mask as .npy files
            np.save(os.path.join(image_out_dir, f"{counter}.npy"), img_patch)
            np.save(os.path.join(mask_out_dir, f"{counter}.npy"), mask_patch)
            counter += 1

    print(f">> Saved {counter} image-mask pairs to {image_out_dir} and {mask_out_dir}")

def get_patches(img, msk, amt=10000, aug=True):    
    if DEBUG_MODE:
        print(f">> get_patches called with: img={repr(img)}, msk={repr(msk)}, amt={repr(amt)}, aug={repr(aug)}")

    is2 = int(1.0 * ISZ)
    xm, ym = img.shape[0] - is2, img.shape[1] - is2

    x, y = [], []

    tr = [0.4, 0.1, 0.1, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001, 0.005]
    for i in range(amt):
        xc = random.randint(0, xm)
        yc = random.randint(0, ym)

        im = img[xc:xc + is2, yc:yc + is2]
        ms = msk[xc:xc + is2, yc:yc + is2]

        for j in range(N_CLS):
            sm = np.sum(ms[:, :, j])
            if 1.0 * sm / is2 ** 2 > tr[j]:
                if aug:
                    if random.uniform(0, 1) > 0.5:
                        im = im[::-1]
                        ms = ms[::-1]
                    if random.uniform(0, 1) > 0.5:
                        im = im[:, ::-1]
                        ms = ms[:, ::-1]

                x.append(im)
                y.append(ms)

    x, y = 2 * np.transpose(x, (0, 3, 1, 2)) - 1, np.transpose(y, (0, 3, 1, 2))
    print(x.shape, y.shape, np.amax(x), np.amin(x), np.amax(y), np.amin(y))
    return x, y


def make_val():    
    if DEBUG_MODE:
        print(f">> make_val called with: ")

    print ("let's pick some samples for validation")
    img = np.load(f'{DATA_DIR}/x_trn_%d.npy' % N_CLS)
    msk = np.load(f'{DATA_DIR}/y_trn_%d.npy' % N_CLS)
    # x, y = get_patches(img, msk, amt=3000)
    x, y = get_patches(img, msk, amt=100)

    np.save(f"{DATA_DIR}/x_tmp_%d" % N_CLS, x)
    np.save(f"{DATA_DIR}/y_tmp_%d" % N_CLS, y)


# def get_unet2():
#     inputs = Input((8, ISZ, ISZ))
#     conv1 = Convolution2D(32, 3,  activation='relu', padding='same')(inputs)
#     conv1 = Convolution2D(32, 3,  activation='relu', padding='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#     conv2 = Convolution2D(64, 3,  activation='relu', padding='same')(pool1)
#     conv2 = Convolution2D(64, 3, activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
#     conv3 = Convolution2D(128, 3,  activation='relu', padding='same')(pool2)
#     conv3 = Convolution2D(128, 3,  activation='relu', padding='same')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

#     conv4 = Convolution2D(256, 3,  activation='relu', padding='same')(pool3)
#     conv4 = Convolution2D(256, 3,  activation='relu', padding='same')(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

#     conv5 = Convolution2D(512, 3,  activation='relu', padding='same')(pool4)
#     conv5 = Convolution2D(512, 3,  activation='relu', padding='same')(conv5)

#     up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
#     conv6 = Convolution2D(256, 3, activation='relu', padding='same')(up6)
#     conv6 = Convolution2D(256, 3,  activation='relu', padding='same')(conv6)

#     up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
#     conv7 = Convolution2D(128, 3,  activation='relu', padding='same')(up7)
#     conv7 = Convolution2D(128, 3,  activation='relu', padding='same')(conv7)

#     up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
#     conv8 = Convolution2D(64, 3,  activation='relu', padding='same')(up8)
#     conv8 = Convolution2D(64, 3,  activation='relu', padding='same')(conv8)

#     up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
#     conv9 = Convolution2D(32, 3,  activation='relu', padding='same')(up9)
#     conv9 = Convolution2D(32, 3,  activation='relu', padding='same')(conv9)

#     conv10 = Convolution2D(N_CLS, 1, 1, activation='sigmoid')(conv9)

#     model = Model(input=inputs, output=conv10)
#     model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
#     return model
    
# def get_unet():
#     inputs = Input((8, ISZ, ISZ))
#     conv1 = Convolution2D(32, 3,  activation='relu', padding='same')(inputs)
#     conv1 = Convolution2D(32, 3,  activation='relu', padding='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#     conv2 = Convolution2D(64, 3,  activation='relu', padding='same')(pool1)
#     conv2 = Convolution2D(64, 3, activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
#     conv3 = Convolution2D(128, 3,  activation='relu', padding='same')(pool2)
#     conv3 = Convolution2D(128, 3,  activation='relu', padding='same')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

# #    conv4 = Convolution2D(256, 3,  activation='relu', padding='same')(pool3)
# #    conv4 = Convolution2D(256, 3,  activation='relu', padding='same')(conv4)
# #    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

# #    conv5 = Convolution2D(512, 3,  activation='relu', padding='same')(pool4)
# #    conv5 = Convolution2D(512, 3,  activation='relu', padding='same')(conv5)

# #    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
# #    conv6 = Convolution2D(256, 3, activation='relu', padding='same')(up6)
# #    conv6 = Convolution2D(256, 3,  activation='relu', padding='same')(conv6)

# #    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
# #    conv7 = Convolution2D(128, 3,  activation='relu', padding='same')(up7)
# #    conv7 = Convolution2D(128, 3,  activation='relu', padding='same')(conv7)

# #    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
# #    conv8 = Convolution2D(64, 3,  activation='relu', padding='same')(up8)
# #    conv8 = Convolution2D(64, 3,  activation='relu', padding='same')(conv8)

# #    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
# #    conv9 = Convolution2D(32, 3,  activation='relu', padding='same')(up9)
# #    conv9 = Convolution2D(32, 3,  activation='relu', padding='same')(conv9)

# #    conv10 = Convolution2D(N_CLS, 1, 1, activation='sigmoid')(conv9)
#     conv10 = Convolution2D(N_CLS, 1, 1, activation='sigmoid')(conv3)

#     model = Model(input=inputs, output=conv10)
#     model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
#     return model





def mask_for_polygons(polygons, im_size):    
    if DEBUG_MODE:
        print(f">> mask_for_polygons called with: polygons={repr(polygons)}, im_size={repr(im_size)}")

    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def mask_to_polygons(mask, epsilon=5, min_area=1.):    
    if DEBUG_MODE:
        print(f">> mask_to_polygons called with: mask={repr(mask)}, epsilon={repr(epsilon)}, min_area={repr(min_area)}")

    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons

## PROBABLY NOT USEFUL, JUST FOR SUBMISSION
# def get_scalers(im_size, x_max, y_min):    
#     if DEBUG_MODE:
#         print(f">> get_scalers called with: im_size={repr(im_size)}, x_max={repr(x_max)}, y_min={repr(y_min)}")

#     # __author__ = Konstantin Lopuhin
#     # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
#     h, w = im_size  # they are flipped so that mask_for_polygons works correctly
#     h, w = float(h), float(w)
#     w_ = 1.0 * w * (w / (w + 1))
#     h_ = 1.0 * h * (h / (h + 1))
#     return w_ / x_max, h_ / y_min


def train_net():
    print ("start train net")
    x_val, y_val = np.load('x_tmp_%d.npy' % N_CLS), np.load('y_tmp_%d.npy' % N_CLS)
    img = np.load('x_trn_%d.npy' % N_CLS)
    msk = np.load('y_trn_%d.npy' % N_CLS)

    x_trn, y_trn = get_patches(img, msk)

    model = get_unet()
#    model.load_weights('weights/unet_10_jk0.7878')
    model_checkpoint = ModelCheckpoint('weights/unet_tmp.hdf5', monitor='loss', save_best_only=True)
    for i in range(1):
        model.fit(x_trn, y_trn, batch_size=64, epochs=1, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint], validation_data=(x_val, y_val))
        del x_trn
        del y_trn
        x_trn, y_trn = get_patches(img, msk)
        score, trs = calc_jacc(model)
        print ('val jk', score)
        model.save_weights('weights/unet_10_jk%.4f' % score)

    return model


def predict_id(id, model, trs):    
    if DEBUG_MODE:
        print(f">> predict_id called with: id={repr(id)}, model={repr(model)}, trs={repr(trs)}")

    img = M(id)
    x = stretch_n(img)

    cnv = np.zeros((960, 960, 8)).astype(np.float32)
    prd = np.zeros((N_CLS, 960, 960)).astype(np.float32)
    cnv[:img.shape[0], :img.shape[1], :] = x

    for i in range(0, 6):
        line = []
        for j in range(0, 6):
            line.append(cnv[i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ])

        x = 2 * np.transpose(line, (0, 3, 1, 2)) - 1
        tmp = model.predict(x, batch_size=4)
        for j in range(tmp.shape[0]):
            prd[:, i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ] = tmp[j]

    # trs = [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1]
    for i in range(N_CLS):
        prd[i] = prd[i] > trs[i]

    return prd[:, :img.shape[0], :img.shape[1]]


def predict_test(model, trs):    
    if DEBUG_MODE:
        print(f">> predict_test called with: model={repr(model)}, trs={repr(trs)}")

    print ("predict test")
    for i, id in enumerate(sorted(set(SB['ImageId'].tolist()))):
        msk = predict_id(id, model, trs)
        np.save('msk/10_%s' % id, msk)
        if i % 100 == 0: print (i, id)


def make_submit():    
    if DEBUG_MODE:
        print(f">> make_submit called with: ")

    print ("make submission file")
    df = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    print (df.head())
    for idx, row in df.iterrows():
        id = row[0]
        kls = row[1] - 1

        msk = np.load('msk/10_%s.npy' % id)[kls]
        pred_polygons = mask_to_polygons(msk)
        x_max = GS.loc[GS['ImageId'] == id, 'Xmax'].as_matrix()[0]
        y_min = GS.loc[GS['ImageId'] == id, 'Ymin'].as_matrix()[0]

        x_scaler, y_scaler = get_scalers(msk.shape, x_max, y_min)

        scaled_pred_polygons = shapely.affinity.scale(pred_polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler,
                                                      origin=(0, 0, 0))

        df.iloc[idx, 2] = shapely.wkt.dumps(scaled_pred_polygons)
        if idx % 100 == 0: print (idx)
    print (df.head())
    df.to_csv('subm/1.csv', index=False)


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


if __name__ == '__main__':
    stick_all_train()
    stick_single_train()
    make_val()
    # print ("--train start")
    # model = train_net()
    # print ("--train end")
    # score, trs = calc_jacc(model)
    # predict_test(model, trs)
    # make_submit()

    # bonus
    # check_predict()
