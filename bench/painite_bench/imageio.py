import cv2
import numpy as np


def read(path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    im = im[..., :3][..., ::-1]
    scale = 65535.0 if im.dtype == np.uint16 else 255.0
    return np.ascontiguousarray(im.astype(np.float32) / scale)


def write(path, img: np.ndarray, bits: int = 8):
    m = 65535 if bits == 16 else 255
    dt = np.uint16 if bits == 16 else np.uint8
    cv2.imwrite(str(path), (np.clip(img, 0, 1)[..., ::-1] * m).round().astype(dt))
