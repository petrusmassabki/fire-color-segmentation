import cv2
import numpy as np


def chen(image, st=175, rt=115):
    """Create a fire color mask according to Chen, Kao and Chang (2003).

    Parameters
    ----------
    image : numpy.ndarray
        Original image (RGB).
    st : int, optional
        Saturation (S channel) threshold.
    rt : int, optional
        Red (R channel) threshold.

    Returns
    -------
    mask : numpy.ndarray
        Fire region binary mask.

    """
    r = image[:, :, 2]
    g = image[:, :, 1]
    b = image[:, :, 0]
    s = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]

    rule_1 = r > rt
    rule_2 = (r >= g) & (g > b)
    rule_3 = s >= (255 - r) * (st / rt)

    mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
    mask[rule_1 & rule_2 & rule_3] = 255

    return mask


def horng(image, h_max=30, s_min=55, v_min=120):
    """Create a fire color mask according to Horng, Peng and Chen (2005).

    Parameters
    ----------
    image : numpy.ndarray
        Original image (RGB).
    h_max : int, optional
        Maximum hue (H channel) threshold.
    s_min : int, optional
        Minimum saturation (S channel) threshold.
    v_min : int, optional
        Minimum value (V channel) threshold.

    Returns
    -------
    mask : numpy.ndarray
        Fire region binary mask.

    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = image_hsv[:, :, 0]
    s = image_hsv[:, :, 1]
    v = image_hsv[:, :, 2]

    rule_1 = (h <= h_max) & (h >= 0)
    rule_2 = (s >= s_min) & (s <= 255)
    rule_3 = (v >= v_min) & (v <= 255)

    mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
    mask[rule_1 & rule_2 & rule_3] = 255

    return mask


def celik(image, t=40):
    """Create a fire color mask according to Çelik and Demirel (2009).

    Parameters
    ----------
    image : numpy.ndarray
        Original image (RGB).
    t : int, optional
        Tau constant determined using a ROC analysis.

    Returns
    -------
    mask : numpy.ndarray
        Fire region binary mask.

    """
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y = image_ycrcb[:, :, 0]
    cr = image_ycrcb[:, :, 1]
    cb = image_ycrcb[:, :, 2]

    f1 = np.poly1d([-6.4000, 947.20])
    f2 = np.poly1d([-0.80953, - 119.81])
    f3 = np.poly1d([-0.32990, 170.23])
    f4 = np.poly1d([4.0000, 996.00])

    (y_mean, cr_mean, cb_mean), _ = cv2.meanStdDev(image_ycrcb)

    rule_1 = y > cb
    rule_2 = cr > cb
    rule_3 = cv2.absdiff(cr, cb) >= t
    rule_4 = (y > y_mean) & (cr > cr_mean) & (cb < cb_mean)
    rule_5 = (cb >= f1(cr)) & (cb > f2(cr)) & (cb <= f3(cr)) & (cb < f4(cr))

    mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
    mask[rule_1 & rule_2 & rule_3 & rule_4 & rule_5] = 255

    return mask


def phillips(image, cp):
    """Create a fire color mask according to Phillips, Shah and Lobo (2002).

    Parameters
    ----------
    image : numpy.ndarray
        Original image (RGB).
    cp : numpy.ndarray
        Color Predicate; a 256 x 256 x 256 binary lookup table array.

    Returns
    -------
    mask : numpy.ndarray
        Fire region binary mask.

    """
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    mask_flat = cp[b.ravel(), g.ravel(), r.ravel()]
    mask = mask_flat.reshape(image.shape[:2])
    mask = np.array(mask, dtype=np.uint8)

    return mask


def backprojection_hsv(image, m, bins=16, threshold=0.4):
    """Create a fire color mask using histogram backprojection in the HSV color space.
    References: Swain and Ballard (1991); Wirth and Zaremba (2010).

    Parameters
    ----------
    image : numpy.ndarray
        OOriginal image (RGB).
    m : numpy.ndarray
        HSV model histogram (3D).
    bins : int, optional
        Number of bins. Must be equal to histogram size.
    threshold : float, optional
        Threshold above which the pixel is considered fire.

    Returns
    -------
    mask : numpy.ndarray
        Fire region binary mask.

    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = np.array(np.floor(image_hsv[:, :, 0] / (180/bins)), dtype=np.uint8)
    s = np.array(np.floor(image_hsv[:, :, 1] / (256/bins)), dtype=np.uint8)
    v = np.array(np.floor(image_hsv[:, :, 2] / (256/bins)), dtype=np.uint8)

    i = cv2.calcHist([image_hsv], [0, 1, 2], None,
                     [bins, bins, bins],
                     [0, 180, 0, 256, 0, 256])
    i = i / np.sum(i)

    with np.errstate(all='ignore'):
        r = np.minimum(np.nan_to_num(np.divide(m, i)), 1)

    b = r[h.ravel(), s.ravel(), v.ravel()]
    b = b.reshape(image.shape[:2])

    mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
    mask[b >= threshold] = 255

    return mask


def backprojection_ycbcr(image, m, bins=16, threshold=0.4):
    """Create a fire color mask using histogram backprojection in the YCbCr color space.
    References: Swain and Ballard (1991); Wirth and Zaremba (2010).

    Parameters
    ----------
    image : numpy.ndarray
        Original image (RGB).
    m : numpy.ndarray
        YCbCr model histogram (3D).
    bins : int, optional
        Number of bins. Must be equal to histogram size.
    threshold : float, optional
        Threshold above which the pixel is considered fire.

    Returns
    -------
    mask : numpy.ndarray
        Fire region binary mask.

    """
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y = np.array(np.floor(image_ycrcb[:, :, 0] / (256/bins)), dtype=np.uint8)
    cr = np.array(np.floor(image_ycrcb[:, :, 1] / (256/bins)), dtype=np.uint8)
    cb = np.array(np.floor(image_ycrcb[:, :, 2] / (256/bins)), dtype=np.uint8)

    i = cv2.calcHist([image_ycrcb], [0, 1, 2], None,
                     [bins, bins, bins],
                     [0, 256, 0, 256, 0, 256])
    i = i / np.sum(i)

    with np.errstate(all='ignore'):
        r = np.minimum(np.nan_to_num(np.divide(m, i)), 1)

    b = r[y.ravel(), cr.ravel(), cb.ravel()]
    b = b.reshape(image.shape[:2])

    mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
    mask[b >= threshold] = 255

    return mask


def rossi(image, constant):
    """Create a fire color mask according to Rossi e Akhloufi (2009).

    Parameters
    ----------
    image : numpy.ndarray
        Original image (RGB).
    constant : float
        Experimentally determined constant.

    Returns
    -------
    mask : numpy.ndarray
        Fire region binary mask.

    """
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    v = image_yuv[:, :, 2]
    v_float = np.float32(v.reshape((-1, 1)))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(v_float, 2, None, criteria,
                                    1, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    v_means_flat = centers[labels.flatten()]
    v_means = v_means_flat.reshape(v.shape)

    _, mask_v = cv2.threshold(v_means, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    (b_m, g_m, r_m), (b_std, g_std, r_std) = cv2.meanStdDev(image, mask=mask_v)
    sigma = max(b_std, g_std, r_std)

    rule = (b - b_m)**2 + (g - g_m)**2 + (r - r_m)**2 <= (constant * sigma)**2

    mask_bgr = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
    mask_bgr[rule] = 255
    mask = cv2.bitwise_and(mask_bgr, mask_v)

    return mask


def rudz(image, hist_blue_ref, hist_green_ref, hist_red_ref):
    """Create a fire color mask according to Rudz et al. (2013).

    Parameters
    ----------
    image : numpy.ndarray
        Original image (RGB).
    hist_blue_ref : numpy.ndarray
        Blue channel reference histogram.
    hist_green_ref : numpy.ndarray
        Green channel reference histogram.
    hist_red_ref : numpy.ndarray
        Red channel reference histogram.

    Returns
    -------
    mask : numpy.ndarray
        Fire region binary mask.

    """
    mean_b_ref = 58.503
    mean_g_ref = 144.522
    mean_r_ref = 225.289
    std_blue_ref = 59.386
    std_green_ref = 73.428
    std_red_ref = 42.953

    tr = 0.317
    tg = 0.108
    tb = 0.191

    cr_const = 0.844
    cg_const = 1.090
    cb_const = 0.828

    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    cb = image_ycrcb[:, :, 2]
    cb_float = np.float32(cb.reshape((-1, 1)))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(cb_float, 4, None, criteria,
                                    2, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    cb_means_flat = centers[labels.flatten()]
    cb_means = cb_means_flat.reshape(cb.shape)

    cb_means_min = np.min(cb_means)

    mask_cb = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
    mask_cb[cb_means == cb_means_min] = 255

    ret, labels_img = cv2.connectedComponents(mask_cb)

    mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)

    for label in range(1, ret):
        blob = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
        blob[labels_img == label] = 255

        if np.count_nonzero(blob) > 256:
            hist_b, _ = np.histogram(image[:, :, 0][blob == 255].ravel(),
                                     bins=256, range=(0, 256), density=True)
            hist_g, _ = np.histogram(image[:, :, 1][blob == 255].ravel(),
                                     bins=256, range=(0, 256), density=True)
            hist_r, _ = np.histogram(image[:, :, 2][blob == 255].ravel(),
                                     bins=256, range=(0, 256), density=True)

            int_b = cv2.norm(hist_blue_ref, hist_b, cv2.NORM_L2)
            int_g = cv2.norm(hist_green_ref, hist_g, cv2.NORM_L2)
            int_r = cv2.norm(hist_red_ref, hist_r, cv2.NORM_L2)

            rule_1 = int_r < tr
            rule_2 = int_g < tg
            rule_3 = int_b < tb
        else:
            diff_r = np.abs(mean_r_ref - np.average(image[:, :, 2][blob == 255]))
            diff_g = np.abs(mean_g_ref - np.average(image[:, :, 1][blob == 255]))
            diff_b = np.abs(mean_b_ref - np.average(image[:, :, 0][blob == 255]))

            rule_1 = diff_r < cr_const * std_red_ref
            rule_2 = diff_g < cg_const * std_green_ref
            rule_3 = diff_b < cb_const * std_blue_ref

        if rule_1 and rule_2 and rule_3:
            mask[labels_img == label] = 255
        else:
            pass

    return mask
