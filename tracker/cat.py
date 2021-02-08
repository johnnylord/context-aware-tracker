import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter

from utils.display import get_color
from track.base import TrackState, TrackAction
from track.utils.kalman3d import chi2inv95
from track.base import BaseTrack
from track.cat import CATTrack

from utils.convert import get_angle_2d
from utils.transform import softmax
from .base import MSVTracker

__all__ = [ "CAT" ]


class CAT(MSVTracker):
    """Tracker that can handle video file of type 'msv'"""

    def __init__(self, max_depth=5, n_depth_levels=10, **kwargs):
        super().__init__(**kwargs)
        # Depth setting
        self.max_depth = max_depth
        self.n_depth_levels = n_depth_levels
        depths = np.linspace(0, max_depth, n_depth_levels)
        starts, ends = depths[:-1], depths[1:]
        self.depth_ranges = np.array([ r for r in zip(starts, ends) ])
        # birdeye_view
        fig, axe = plt.subplots(figsize=(16, 8))
        self.fig = fig
        self.axe = axe
        # tracks in system
        self.tracks = []
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        """Return video frames & current track set"""
        self.start_time = time.time()

        frames = self.video.read()
        if frames['video'] is None:
            raise StopIteration

        # Extract observations and filter out anamoly ones
        # ====================================================================
        bodyposes = frames['bodyposes']
        if len(bodyposes) != 0:
            masks = []
            observations = []
            for pose in bodyposes:
                mask = pose['mask']
                bbox = pose['bbox']
                # Filter operation
                area = (bbox[3]-bbox[1])*(bbox[2]-bbox[0])
                if area < 68100 or area > 543000:
                    continue
                feature = pose['feature']
                keypoints = pose['keypoints']
                observation = np.concatenate([
                                            bbox.reshape(-1),
                                            keypoints.reshape(-1),
                                            feature.reshape(-1)])
                masks.append(mask)
                observations.append(observation)
            observations = np.array(observations)
        else:
            masks = []
            observations = np.array([])

        # No observation at all
        if len(observations) == 0:
            # Propage track state
            for t in self.tracks:
                t.predict()
                t.miss()
            # Remove dead track
            self.tracks = [ t for t in self.tracks if t.state != TrackState.DEAD ]
            # Return current state
            tracks = [  t.content
                        for t in self.tracks
                        if t.state == TrackState.TRACKED ]
            return frames, tracks

        # Extract observations (N, 208)
        # ======================================================================
        # - The first four fields is bounding boxes with confidence score
        # - The next 75 fields are reshaped from (25, 3) matrix representing
        #   25 key joints (x, y, conf)
        # - The next 128 fields are reid feature vector of each object
        masks = masks
        bboxes = observations[:, :5]
        features = observations[:, 5+75:]
        keypointss = observations[:, 5:5+75]

        # Augment bboxes with depth values
        # (replace bboxes[:, 5] with depth value, instead of conf value)
        depth_frame = frames['depth']
        dboxes = self._assign_depth(bboxes, masks, depth_frame)
        observations = np.concatenate([ dboxes, keypointss, features ], axis=1)

        # Propage track state
        for t in self.tracks:
            t.predict()

        # Split visible tracks based on tracking state
        # ============================================================================
        lost_tracks = [ t for t in self.tracks if t.state == TrackState.LOST ]
        tracked_tracks = [ t for t in self.tracks if t.state == TrackState.TRACKED ]
        tentative_tracks = [ t for t in self.tracks if t.state == TrackState.TENTATIVE ]

        # Perform association
        # ==================================================================
        match_pairs = []
        unmatch_tracks = []

        # Perform association on visible tracked tracks
        pairs, tracks, observations = self._matching_cascade_time(tracked_tracks,
                                                                observations,
                                                                mode='maha_cos',
                                                                threshold=0.6)
        match_pairs.extend(pairs)
        unmatch_tracks.extend(tracks)

        # Perform association on visible lost tracks
        pairs, tracks, observations = self._matching_cascade_time(lost_tracks,
                                                                observations,
                                                                mode='cos',
                                                                threshold=0.4)
        match_pairs.extend(pairs)
        unmatch_tracks.extend(tracks)

        # Perform association on visible tentative tracks
        pairs, tracks, observations = self._matching_cascade_time(tentative_tracks,
                                                                observations,
                                                                mode='maha_iou',
                                                                threshold=0.5)
        match_pairs.extend(pairs)
        unmatch_tracks.extend(tracks)

        # Update tracks states
        # ==================================================================
        for pair in match_pairs:
            # Extract observation
            track = pair[0]
            dbox, feature = pair[1][:5], pair[1][5+75:]
            keypoints = pair[1][5:5+75].reshape(25, 3)
            angle = get_angle_2d(keypoints)[0]
            # Update track
            track.register(feature, angle)
            track.update(dbox)
            track.hit()

        for track in unmatch_tracks:
            track.miss()

        # Create newborn tracks
        # ==================================================================
        for observation in observations:
            # Extract observation
            dbox, feature = observation[:5], observation[5+75:]
            keypoints = observation[5:5+75].reshape(25, 3)
            angle = get_angle_2d(keypoints)[0]
            # Create track
            track = CATTrack(dbox=dbox,
                            angle=angle,
                            feature=feature,
                            id=self.counter)
            self.tracks.append(track)
            self.counter += 1

        self.end_time = time.time()

        # Remove dead track
        self.tracks = [ t for t in self.tracks if t.state != TrackState.DEAD ]

        # Return current state
        tracks = [ t.content for t in self.tracks if t.state == TrackState.TRACKED ]
        return frames, tracks

    def _matching_cascade_depth(self, tracks, observations, mode='cos', threshold=0.4, **kwargs):
        all_pairs = []
        all_tracks = []
        # Perform matching cascade association with respect to depth
        for depth_range in self.depth_ranges:
            depth_start, depth_end = depth_range
            depth_tracks = [ t
                            for t in tracks
                            if depth_start <= t.content['depth'] < depth_end ]
            if len(depth_tracks) == 0:
                continue
            pairs, umtracks, observations = self._matching_cascade_time(depth_tracks,
                                                                        observations,
                                                                        mode=mode,
                                                                        threshold=threshold,
                                                                        **kwargs)
            all_pairs.extend(pairs)
            all_tracks.extend(umtracks)
        return all_pairs, all_tracks, observations

    def _matching_cascade_time(self, tracks, observations, mode='cos', threshold=0.4, **kwargs):
        all_pairs = []
        all_tracks = []
        # Perform matching cascade association with respect to time
        for priority_level in range(BaseTrack.MAX_PRIORITY, 0, -1):
            priority_tracks = [ t for t in tracks if t.priority == priority_level ]
            if len(priority_tracks) == 0:
                continue
            pairs, umtracks, observations = self._associate(priority_tracks,
                                                            observations,
                                                            mode=mode,
                                                            threshold=threshold,
                                                            **kwargs)
            all_pairs.extend(pairs)
            all_tracks.extend(umtracks)
        return all_pairs, all_tracks, observations

    def _associate(self,
                tracks,
                observations,
                mode='cos',
                threshold=0.4,
                n_degrees=3,
                use_orient_pool=False):
        """Perfrom tracking association"""
        if len(tracks) == 0 and len(observations) != 0:
            return [], [], observations
        elif len(tracks) != 0 and len(observations) == 0:
            return [], tracks, []
        elif len(tracks) == 0 and len(observations) == 0:
            return [], [], []

        dboxes = observations[:, :5]
        features = observations[:, 5+75:]
        keypointss = observations[:, 5:5+75]
        angles = np.array([ get_angle_2d(ks.reshape(25, 3))[0] for ks in keypointss ])

        # Concstruct cost matrix
        if mode == 'iou':
            cost_mat = np.array([ t.iou_dist(dboxes) for t in tracks ])
        elif mode == 'cos':
            cost_mat = np.array([ t.cos_dist(features, angles) for t in tracks ])
        elif mode == 'maha_iou':
            prob_iou = 1 - np.array([ t.iou_dist(dboxes) for t in tracks ])
            prob_maha = np.array([ -t.square_maha_dist(dboxes, n_degrees=n_degrees) for t in tracks ])
            prob_maha = np.array([ softmax(row) for row in prob_maha ])
            cost_mat = 1 - prob_iou * prob_maha
        elif mode == 'maha_cos':
            prob_cos = 1 - np.array([ t.cos_dist(features, angles) for t in tracks ])
            prob_maha = np.array([ -t.square_maha_dist(dboxes, n_degrees=n_degrees) for t in tracks ])
            prob_maha = np.array([ softmax(row) for row in prob_maha ])
            cost_mat = 1 - prob_cos * prob_maha

        # Filter out impossible entry
        mask_mat = np.array([ t.square_maha_dist(dboxes, n_degrees=3) for t in tracks ])
        cost_mat[mask_mat > chi2inv95[3]] = 10000

        # Perform greedy matching algorithm
        tindices, oindices = linear_sum_assignment(cost_mat)
        match_pairs = [ pair
                        for pair in zip(tindices, oindices)
                        if cost_mat[pair[0], pair[1]] <= threshold ]

        # Prepare matching result
        pairs = [ (tracks[pair[0]], observations[pair[1]]) for pair in match_pairs ]

        unmatch_tindices = set(range(len(tracks))) - set([ pair[0] for pair in match_pairs ])
        unmatch_tindices = sorted(list(unmatch_tindices))
        unmatch_tracks = [ tracks[i] for i in unmatch_tindices ]

        unmatch_oindices = set(range(len(observations))) - set([ pair[1] for pair in match_pairs ])
        unmatch_oindices = sorted(list(unmatch_oindices))
        unmatch_observations = [ observations[i] for i in unmatch_oindices ]

        return pairs, unmatch_tracks, np.array(unmatch_observations)

    def _assign_depth(self, bboxes, masks, depth_frame):
        """Assign depth value to bboxes"""
        # Convert depth_frame to depth_map (RGB -> depth in meter)
        depth_map = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
        depth_map = ((255-depth_map)/255)*self.max_depth

        # Compute depth value for each object
        for i, (bbox, mask) in enumerate(zip(bboxes, masks)):
            xmin, ymin = int(bbox[0]), int(bbox[1])
            xmax, ymax = int(bbox[2]), int(bbox[3])
            bool_mask = mask >= 200
            crop_map = depth_map[ymin:ymax, xmin:xmax]
            crop_map = crop_map[bool_mask]
            depth = np.mean(crop_map[crop_map > 0])
            bboxes[i, 4] = depth

        return bboxes

    def _is_occluded(self, target):
        """Return whether the track is occluded by other track or not"""
        # Compute target track center
        tracked_tracks = [ t for t in self.tracks if t.state == TrackState.TRACKED ]
        for track in tracked_tracks:
            if (
                track.id == target.id
                or track.content['depth'] > target.content['depth']
            ):
                continue
            # Compute overlap
            xmin1, _, xmax1, _ = target.content['bbox']
            xmin2, _, xmax2, _ = track.content['bbox']
            overlap = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
            ratio = overlap/(xmax1-xmin1)
            if ratio >= 0.5:
                return True

        return False

    @property
    def bird_view(self):
        """Generate birdeye view of tracked tracks"""
        width = self.video.width
        height = self.video.height
        for track in self.tracks:
            if track.state == TrackState.TENTATIVE:
                continue
            occluded = self._is_occluded(track)
            content = track.content
            bbox = content['bbox']
            depth = content['depth']
            xyz = content['mean'][:3]
            std = np.sqrt(content['covar'][:3, :3])

            # Plotting metadata
            text = f"ID:{content['id']}" + ("(occluded)" if occluded else "")
            text += f"\nPriority: {track.priority}"
            if occluded:
                color = (np.array([0, 0, 255])/255).tolist()
            else:
                color = (np.array(get_color(content['id']))/255).tolist()
            color = tuple(color[::-1])
            # Plot confidence distribution
            mean_x = xyz[0]
            mean_z = xyz[2]
            scale_x = std[0, 0]*3
            scale_z = std[2, 2]*3
            ellipse = Ellipse((0, 0), width=2, height=2, facecolor=color, alpha=0.4)
            transf = transforms.Affine2D() \
                        .rotate_deg(45) \
                        .scale(scale_x, scale_z) \
                        .translate(mean_x, mean_z)
            ellipse.set_transform(transf+self.axe.transData)
            self.axe.add_patch(ellipse)
            # Plot horizantol bar
            bar_x = np.linspace(bbox[0], bbox[2])
            bar_y = np.ones_like(bar_x)*depth
            self.axe.plot(bar_x, bar_y, color=color, lw=5, alpha=0.6)
            # Plot end points
            self.axe.scatter(bbox[0], depth, s=100, color=color)
            self.axe.scatter(bbox[2], depth, s=100, color=color)
            # Plot text
            self.axe.text((bbox[0]+bbox[2])/2, depth, s=text,
                        ha='center', va='center',
                        fontsize=14, fontweight='bold',
                        color='white', bbox=dict(facecolor=color))
        # Convert matplot canvas to numpy frame
        self.axe.set_xlabel("Position (pixel)")
        self.axe.set_ylabel("Depth (meter)")
        self.axe.set_xticks(np.linspace(0, width, 10))
        self.axe.set_yticks(np.linspace(0, self.max_depth, self.n_depth_levels))
        self.axe.set_xlim(0, width)
        self.axe.set_ylim(0, self.max_depth)
        self.axe.grid(which='minor', lw=3, alpha=0.3)
        self.axe.grid(which='major', lw=3, alpha=0.5)
        self.fig.canvas.draw()
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(self.fig.canvas.get_width_height()[::-1]+(3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.axe.clear()
        return img
