import cv2
import numpy as np
import pydicom
from config import IMPORTANT_DICOM_TAGS

def dicom_to_rgb(ds):
    try:
        img = ds.pixel_array.astype(float)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        slope = getattr(ds, 'RescaleSlope', 1)
        img = img * slope + intercept
        center = ds.WindowCenter[0] if isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else ds.WindowCenter
        width = ds.WindowWidth[0] if isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else ds.WindowWidth
        low, high = center - width // 2, center + width // 2
        img = np.clip(img, low, high)
        img = (img - low) / (high - low) * 255.0
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    except Exception as e:
        print(f"Ошибка чтения DICOM: {e}")
        return np.zeros((512, 512, 3), dtype=np.uint8)

def clean_num(val):
    try:
        if isinstance(val, (list, pydicom.multival.MultiValue)):
            return ", ".join([f"{float(x):.3f}" for x in val])
        s = str(val).replace('[','').replace(']','').split(',')[0]
        return f"{float(s):.3f}".rstrip('0').rstrip('.')
    except:
        return str(val)

def extract_meta(ds):
    meta = {}
    for tag in IMPORTANT_DICOM_TAGS:
        meta[tag] = clean_num(getattr(ds, tag, "Н/Д"))
    return meta