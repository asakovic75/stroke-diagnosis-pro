import cv2
import numpy as np

def get_hu_analysis(image_orig, mask_256, ds):
    try:
        mask_orig = cv2.resize(mask_256, (image_orig.shape[1], image_orig.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        raw_pixels = ds.pixel_array.astype(float)
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        hu_data = raw_pixels * slope + intercept
        stroke_pixels = hu_data[mask_orig > 0]
        if len(stroke_pixels) > 0:
            avg_val = np.mean(stroke_pixels)
            label = "Кровь/Геморрагия" if avg_val > 50 else "Ишемия/Отек"
            return f"{avg_val:.1f} HU ({label})"
    except Exception as e:
        print(f"Ошибка анализа плотности: {e}")
    return "Н/Д"

def get_artery_basin(mask_256, side_ru):
    if side_ru == "Не выявлено":
        return side_ru
    y_coords, _ = np.where(mask_256 > 0)
    if len(y_coords) == 0:
        return side_ru
    mean_y = np.mean(y_coords)
    if mean_y < 80:
        basin = "бассейн ПМА (передняя)"
    elif mean_y > 180:
        basin = "бассейн ЗМА (задняя)"
    else:
        basin = "бассейн СМА (средняя)"
    return f"{side_ru} полушарие, {basin}"

def get_side_and_basin(mask):
    mask_sum_left = np.sum(mask[:, :128])
    mask_sum_right = np.sum(mask[:, 128:])
    raw_side = "Левое" if mask_sum_left > mask_sum_right else "Правое"
    side_ru = get_artery_basin(mask, raw_side) if np.sum(mask) > 0 else "Не выявлено"
    return raw_side, side_ru

def calculate_area_percent(mask):
    mask_sum = np.sum(mask)
    return (mask_sum / (256 * 256)) * 100

def compute_confidence(prob, mask):
    mask_sum = np.sum(mask)
    if mask_sum > 0:
        return np.mean(prob[mask > 0]) * 100
    return 100.0