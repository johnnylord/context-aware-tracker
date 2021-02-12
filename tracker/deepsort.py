import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.optimize import linear_sum_assignment

from utils.display import get_color
from track.base import TrackState, TrackAction
from track.utils.kalman2d import chi2inv95
from track.deepsort import DeepTrack
from track.base import BaseTrack

from .base import MSVTracker

__all__ = [ "DeepSORT" ]


class DeepSORT(MSVTracker):
    """Tracker that can handle video file of type 'msv'"""

    def __init__(self, max_depth=4, n_depth_levels=5, **kwargs):
        super().__init__(**kwargs)
        # Virtual depth
        self.max_depth = max_depth
        self.n_depth_levels = n_depth_levels
        # birdeye_view
        fig, axe = plt.subplots(figsize=(16, 8))
        self.fig = fig
        self.axe = axe

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

        # Extract metadata
        bodyposes = frames['bodyposes']
        if len(bodyposes) != 0:
            observations = []
            for pose in bodyposes:
                mask = pose['mask']
                bbox = pose['bbox']
                area = (bbox[3]-bbox[1])*(bbox[2]-bbox[0])
                if area < 68100 or area > 543000:
                    continue
                feature = pose['feature']
                keypoints = pose['keypoints']
                observation = np.concatenate([
                                            bbox.reshape(-1),
                                            keypoints.reshape(-1),
                                            feature.reshape(-1)])
                observations.append(observation)
            observations = np.array(observations)
        else:
            observations = np.array([])

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

        bboxes = observations[:, :5]
        features = observations[:, 5+75:]
        observations = np.concatenate([ bboxes, features ], axis=1)

        # Propage track state
        for t in self.tracks:
            t.predict()

        # Split tracks by state
        confirm_tracks = [ t for t in self.tracks if t.state != TrackState.TENTATIVE ]
        tentative_tracks = [ t for t in self.tracks if t.state == TrackState.TENTATIVE ]

        # Perfrom association
        match_pairs = []
        unmatch_tracks = []

        # Confirm tracks association
        pairs, tracks, observations = self._matching_cascade(confirm_tracks, observations, mode='cos', threshold=0.4)
        match_pairs.extend(pairs)
        unmatch_tracks.extend(tracks)

        # Tentative tracks association
        pairs, tracks, observations = self._associate(tentative_tracks, observations, mode='iou', threshold=0.3)
        match_pairs.extend(pairs)
        unmatch_tracks.extend(tracks)

        # Update matching track set
        for pair in match_pairs:
            track, bbox, feature = pair[0], pair[1][:4], pair[1][5:]
            track.register(feature)
            track.update(bbox)
            track.hit()

        # Update unmathcing track set
        for track in unmatch_tracks:
            track.miss()

        # Create new track
        for observation in observations:
            bbox = observation[:4]
            feature = observation[5:]
            track = DeepTrack(bbox=bbox, feature=feature, id=self.counter)
            self.tracks.append(track)
            self.counter += 1

        self.end_time = time.time()

        # Remove dead track
        self.tracks = [ t for t in self.tracks if t.state != TrackState.DEAD ]

        # Return current state
        tracks = [ t.content for t in self.tracks if t.state == TrackState.TRACKED ]
        return frames, tracks

    def _matching_cascade(self, confirm_tracks, observations, mode='cos', threshold=0.5):
        all_pairs = []
        all_tracks = []
        # Perform matching cascade association
        for priority_level in range(BaseTrack.MAX_PRIORITY, 0, -1):
            priority_tracks = [ t for t in confirm_tracks if t.priority == priority_level ]
            if len(priority_tracks) == 0:
                continue
            pairs, tracks, observations = self._associate(priority_tracks,
                                                        observations,
                                                        mode=mode,
                                                        threshold=threshold)
            all_pairs.extend(pairs)
            all_tracks.extend(tracks)
        return all_pairs, all_tracks, observations

    def _associate(self, tracks, observations, mode='cos', threshold=0.5):
        if len(tracks) == 0 and len(observations) != 0:
            return [], [], observations
        elif len(tracks) != 0 and len(observations) == 0:
            return [], tracks, []
        elif len(tracks) == 0 and len(observations) == 0:
            return [], [], []

        bboxes = observations[:, :4]
        features = observations[:, 5:]

        # Concstruct cost matrix
        if mode == 'cos':
            cost_mat = np.array([ t.cos_dist(features) for t in tracks ])
        elif mode == 'iou':
            cost_mat = np.array([ t.iou_dist(bboxes) for t in tracks ])
        mask_mat = np.array([ t.square_maha_dist(bboxes) for t in tracks ])
        cost_mat[mask_mat > chi2inv95[2]] = 10000

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

    @property
    def bird_view(self):
        """Generate birdeye view of tracked tracks"""
        width = self.video.width
        height = self.video.height
        for track in self.tracks:
            if track.state == TrackState.TENTATIVE:
                continue
            content = track.content
            bbox = content['bbox']
            xy = content['mean'][:2]
            std = np.sqrt(content['covar'][:2, :2])

            # Plotting metadata
            text = f"ID:{content['id']}"
            text += f"\nPriority: {track.priority}"
            color = (np.array(get_color(content['id']))/255).tolist()
            color = tuple(color[::-1])
            # Plot confidence distribution
            mean_x = xy[0]
            mean_z = 2
            depth = mean_z
            scale_x = std[0, 0]*3
            scale_z = 0.1*3
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
