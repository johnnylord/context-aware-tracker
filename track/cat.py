import scipy
import numpy as np
from .base import BaseTrack, TrackState
from .utils.kalman3d import KalmanFilter3D
from .utils.convert import tlbr_to_xyah, xyah_to_tlbr

__all__ = [ "CATTrack" ]

class CATTrack(BaseTrack):
    """A semantic and spatial track modeled with 3D kalman filter"""
    BINS = {
        'positive': [
            (0, 45),
            (45, 135),
            (135, 180),
            (-1, -1),
        ],
        'negative': [
            (315, 360),
            (-1, -1),
            (180, 225),
            (225, 315),
        ]
    }
    POOL_SIZE = 30
    def __init__(self, dbox, feature, angle, **kwargs):
        super().__init__(**kwargs)
        self.kf = KalmanFilter3D()

        # Initialize kalman state
        xyah = np.array(tlbr_to_xyah(dbox[:4]))
        xyah = xyah.tolist()
        xyah.insert(2, dbox[4])
        xyzah = np.array(xyah)
        self.mean, self.covar = self.kf.initiate(xyzah)

        # Initialize feature pool
        self.bin_pools = dict([ (i, []) for i in range(len(CATTrack.BINS['positive'])) ])
        bin_idx = self._determine_bin(angle)
        self.bin_pools[bin_idx].append(feature)
        self.time_pool = [ feature ]

    @property
    def content(self):
        xyzah = self.mean[:5]
        xyah = xyzah.tolist()
        depth = xyah.pop(2)
        bbox = xyah_to_tlbr(xyah)

        content = {
            'id': self.id,
            'bbox': bbox,
            'depth': depth,
            'mean': self.mean[:5],
            'covar': self.covar[:5, :5],
            'priority': self.priority,
            'bin_pools': self.bin_pools,
            'features': self.time_pool
            }
        return content

    def predict(self, position_only=False, covar_only=False):
        mean, covar = self.kf.predict(self.mean, self.covar)

        if covar_only:
            self.covar = covar
            return self.mean, self.covar

        if not position_only:
            self.mean = mean
            self.covar = covar
        else:
            self.mean[:3] = mean[:3]
            self.covar = covar

        return self.mean, self.covar

    def update(self, dbox):
        xyah = np.array(tlbr_to_xyah(dbox[:4]))
        xyah = xyah.tolist()
        xyah.insert(2, dbox[4])
        xyzah = np.array(xyah)
        self.mean, self.covar = self.kf.update(mean=self.mean,
                                                covariance=self.covar,
                                                observation=xyzah)
        return self.mean, self.covar

    def register(self, feature, angle):
        # Add feature to oriented pool
        bin_idx = self._determine_bin(angle)
        self.bin_pools[bin_idx].append(feature)
        if len(self.bin_pools[bin_idx]) > CATTrack.POOL_SIZE:
            self.bin_pools[bin_idx] = self.bin_pools[bin_idx][1:]
        # Add feature to time pool
        self.time_pool.append(feature)
        if len(self.time_pool) > CATTrack.POOL_SIZE:
            self.time_pool = self.time_pool[1:]

    def iou_dist(self, dboxes):
        """Return iou distance between track and bboxes

        Args:
            dboxes (np.ndarray): array of shape (N, 5)

        Return:
            A N dimensional iou distance vector

        Note:
            A dbox is (xmin, ymin, xmax, ymax, depth)
        """
        xyah = self.mean[:5].tolist()
        xyah.pop(2)
        xyah = np.array(xyah)
        bbox = np.array([xyah_to_tlbr(xyah)])
        x11, y11, x12, y12 = np.split(bbox, 4, axis=1)
        x21, y21, x22, y22 = np.split(dboxes[:, :4], 4, axis=1)

        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))

        interArea = np.maximum((xB-xA+1), 0)*np.maximum((yB-yA+1), 0)
        bbox1Area = (x12-x11+1)*(y12-y11+1)
        bbox2Area = (x22-x21+1)*(y22-y21+1)

        iou = interArea / (bbox1Area+np.transpose(bbox2Area)-interArea)
        return (1 - iou).reshape(-1)

    def cos_dist(self, features, angles, use_orient_pool=False):
        """Return cosine distance between features and feature_pool of track

        Args:
            features (np.ndarray): array of shape (N, n_features)
            angles (np.ndarray): array of shape (N,)

        Return:
            A N dimensional cosine distance vector

        Note:
            Each feature vector is a unit vector
        """
        outputs = []
        for feature, angle in zip(features, angles):
            if use_orient_pool:
                bin_idx = self._determine_bin(angle)
                if len(self.bin_pools[bin_idx]) >= (CATTrack.POOL_SIZE // 3 * 2):
                    feature_pool = np.array(self.bin_pools[bin_idx])
                else:
                    feature_pool = np.array(self.time_pool)
            else:
                feature_pool = np.array(self.time_pool)
            cosines = np.dot(feature_pool, feature.T)
            cosines[cosines < 0] = 0
            cosine = (1 - cosines).mean(axis=0)
            outputs.append(cosine)

        return np.array(outputs)

    def square_maha_dist(self, dboxes, n_degrees=3):

        """Return squared mahalonobius distance between track and bboxes

        Args:
            bboxes (np.ndarray): array of shape (N, 4)

        Return:
            A N dimensional distance vector

        Note:
            A bbox is (xmin, ymin, xmax, ymax)
        """
        # Normalize data
        xyahs = np.array([ tlbr_to_xyah(dbox[:4]) for dbox in dboxes ]).tolist()
        for xyah, depth in zip(xyahs, dboxes[:, 4]):
            xyah.insert(2, depth)
        xyzahs = np.array(xyahs)
        mean, covar = self.kf._project(self.mean, self.covar)

        # Align number of dimensions
        mean, covar = mean[:n_degrees], covar[:n_degrees, :n_degrees]
        xyzahs = xyzahs[:, :n_degrees]

        # Apply mahalonobius distance formula
        cholesky_factor = np.linalg.cholesky(covar)
        d = xyzahs - mean
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T,
                                        check_finite=False,
                                        overwrite_b=True,
                                        lower=True)
        squared_maha = np.sum(z*z, axis=0)
        return squared_maha

    def _determine_bin(self, angle):
        if 0 < angle <= 180:
            BINS = CATTrack.BINS['positive']
        else:
            BINS = CATTrack.BINS['negative']

        for i, bins in enumerate(BINS):
            if bins[0] < angle <= bins[1]:
                return i

        raise RuntimeError("Cannot determine bin with angle {:.2f}".format(angle))
