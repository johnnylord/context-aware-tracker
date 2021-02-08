import time
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from track.base import TrackState, TrackAction
from track.utils.kalman2d import chi2inv95
from track.sort import SORTTrack

from .base import MSVTracker

__all__ = [ "SORT" ]


class SORT(MSVTracker):
    """Tracker that can handle video file of type 'msv'"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
                observations.append(bbox)
            observations = np.array(observations)
        else:
            observations = np.array([])

        if len(observations) == 0:
            # Propage track state
            for t in self.tracks:
                t.miss()
                t.predict()
            # Remove dead track
            self.tracks = [ t for t in self.tracks if t.state != TrackState.DEAD ]
            # Return current state
            tracks = [  t.content
                        for t in self.tracks
                        if t.state == TrackState.TRACKED ]
            return frames, tracks

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
        pairs, tracks, observations = self._associate(confirm_tracks, observations)
        match_pairs.extend(pairs)
        unmatch_tracks.extend(tracks)

        # Tentative tracks association
        pairs, tracks, observations = self._associate(tentative_tracks, observations)
        match_pairs.extend(pairs)
        unmatch_tracks.extend(tracks)

        # Update matching track set
        for pair in match_pairs:
            track, bbox = pair[0], pair[1][:4]
            track.update(bbox)
            track.hit()

        # Update unmathcing track set
        for track in unmatch_tracks:
            track.miss()

        # Create new track
        for observation in observations:
            bbox = observation[:4]
            track = SORTTrack(bbox=bbox, id=self.counter)
            self.tracks.append(track)
            self.counter += 1

        self.end_time = time.time()

        # Remove dead track
        self.tracks = [ t for t in self.tracks if t.state != TrackState.DEAD ]

        # Return current state
        tracks = [ t.content for t in self.tracks if t.state == TrackState.TRACKED ]
        return frames, tracks

    def _associate(self, tracks, observations, threshold=0.4):
        if len(tracks) == 0 and len(observations) != 0:
            return [], [], observations
        elif len(tracks) != 0 and len(observations) == 0:
            return [], tracks, []
        elif len(tracks) == 0 and len(observations) == 0:
            return [], [], []

        bboxes = observations[:, :4]

        # Concstruct cost matrix
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
