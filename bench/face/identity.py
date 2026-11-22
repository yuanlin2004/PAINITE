"""Identity similarity: ArcFace (insightface buffalo_l w600k_r50, ONNX on CPU) cosine between a
restored face and its ground truth, both cropped to 112 px with the same 5 landmarks (detected
once on the ground truth), so only the face content differs.
"""
import cv2
import numpy as np
import onnxruntime as ort

from restorers import FACE_WEIGHTS

# insightface arcface_dst, the 112x112 ArcFace template
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


class ArcFace:
    def __init__(self):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 8
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(FACE_WEIGHTS / "buffalo_l/w600k_r50.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    @staticmethod
    def crop(img_bgr: np.ndarray, lm5: np.ndarray) -> np.ndarray:
        m = cv2.estimateAffinePartial2D(lm5.astype(np.float32), ARCFACE_DST, method=cv2.LMEDS)[0]
        return cv2.warpAffine(img_bgr, m, (112, 112), borderValue=0)

    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        x = np.stack([c[..., ::-1] for c in crops_bgr]).astype(np.float32)
        x = ((x - 127.5) / 127.5).transpose(0, 3, 1, 2)
        e = np.concatenate([self.sess.run(None, {self.inp: x[i:i + 1]})[0] for i in range(len(x))])  # batch dim is fixed at 1
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def similarity(self, a_bgr, b_bgr, lm5) -> float:
        ea, eb = self.embed([self.crop(a_bgr, lm5), self.crop(b_bgr, lm5)])
        return float(ea @ eb)
