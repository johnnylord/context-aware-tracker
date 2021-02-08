import scipy
import numpy as np
from .base import BaseTrack
from .utils.kalman2d import KalmanFilter2D
from .utils.convert import tlbr_to_xyah, xyah_to_tlbr

__all__ = [ "DeepTrack" ]

class DeepTrack(BaseTrack):
    """A semantic and spatial track modeled with 2D kalman filter"""
    POOL_SIZE = 30

    def __init__(self, bbox, feature, **kwargs):
        super().__init__(**kwargs)
        self.kf = KalmanFilter2D()
        # Initialize kalman state
        xyah = tlbr_to_xyah(bbox)
        self.mean, self.covar = self.kf.initiate(xyah)

        # Initialize feature pool
        self.feature_pool = [ feature ]

    @property
    def content(self):
        xyah = self.mean[:4]
        bbox = xyah_to_tlbr(xyah)

        content = {
            'id': self.id,
            'bbox': bbox,
            'mean': self.mean[:4],
            'covar': self.covar[:4, :4],
            'priority': self.priority,
            'features': self.feature_pool
            }
        return content

    def predict(self, position_only=False, preserve_covar=False):
        mean, covar = self.kf.predict(self.mean, self.covar)

        if not position_only:
            self.mean = mean
        else:
            self.mean[:2] = mean[:2]

        if not preserve_covar:
            self.covar = covar

        return self.mean, self.covar

    def update(self, bbox, position_only=False):
        xyah = tlbr_to_xyah(bbox)
        mean, covar = self.kf.update(mean=self.mean,
                                    covariance=self.covar,
                                    observation=xyah)
        if not position_only:
            self.mean = mean
        else:
            self.mean[:2] = mean[:2]
        self.covar = covar

        return self.mean, self.covar

    def register(self, feature):
        self.feature_pool.append(feature)
        if len(self.feature_pool) >= DeepTrack.POOL_SIZE:
            self.feature_pool = self.feature_pool[1:]

    def iou_dist(self, bboxes):
        """Return iou distance between track and bboxes

        Args:
            bboxes (np.ndarray): array of shape (N, 4)

        Return:
            A N dimensional iou distance vector

        Note:
            A bbox is (xmin, ymin, xmax, ymax)
        """
        bbox = np.array([xyah_to_tlbr(self.mean[:4])])
        x11, y11, x12, y12 = np.split(bbox, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes, 4, axis=1)

        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))

        interArea = np.maximum((xB-xA+1), 0)*np.maximum((yB-yA+1), 0)
        bbox1Area = (x12-x11+1)*(y12-y11+1)
        bbox2Area = (x22-x21+1)*(y22-y21+1)

        iou = interArea / (bbox1Area+np.transpose(bbox2Area)-interArea)
        return (1 - iou).reshape(-1)

    def square_maha_dist(self, bboxes, n_degrees=2):
        """Return squared mahalonobius distance between track and bboxes

        Args:
            bboxes (np.ndarray): array of shape (N, 4)

        Return:
            A N dimensional distance vector

        Note:
            A bbox is (xmin, ymin, xmax, ymax)
        """
        # Normalize data
        xyahs = np.array([ tlbr_to_xyah(bbox) for bbox in bboxes ])
        mean, covar = self.kf._project(self.mean, self.covar)

        # Align number of dimensions
        mean, covar = mean[:n_degrees], covar[:n_degrees, :n_degrees]
        xyahs = xyahs[:, :n_degrees]

        # Apply mahalonobius distance formula
        cholesky_factor = np.linalg.cholesky(covar)
        d = xyahs - mean
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T,
                                        check_finite=False,
                                        overwrite_b=True,
                                        lower=True)
        squared_maha = np.sum(z*z, axis=0)

        return squared_maha

    def cos_dist(self, features):
        """Return cosine distance between features and feature_pool of track

        Args:
            features (np.ndarray): array of shape (N, n_features)

        Return:
            A N dimensional cosine distance vector

        Note:
            Each feature vector is a unit vector
        """
        feature_pool = np.array(self.feature_pool)
        cosines = np.dot(feature_pool, features.T)
        return (1 - cosines).mean(axis=0)
