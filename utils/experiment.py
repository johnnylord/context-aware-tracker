import os
import os.path as osp

import numpy as np
import pickle


__all__ = [ "TrackAccumulator" ]

class TrackAccumulator:

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.snapshots = []
        self.snapshots_pools = []
        self.results = []
        if not osp.exists(output_dir):
            os.makedirs(output_dir)

    def accumulate(self, fid, tracks):
        snapshot = {}
        snapshot_pools = {}
        for t in sorted(tracks, key=lambda t: int(t['id'])):
            # MOT2D
            tid, bbox = t['id'], t['bbox']
            bb_left, bb_top = bbox[0], bbox[1]
            bb_width, bb_height = bbox[2]-bbox[0], bbox[3]-bbox[1]
            # MOT3D
            depth = t['depth'] if 'depth' in t else None
            # Custom experiment
            quota = t['quota'] if 'quota' in t else None
            features = t['features'] if 'features' in t else None
            bin_pools = t['bin_pools'] if 'bin_pools' in t else None
            # Export metadata
            result = f"{fid},{tid},{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},-1,-1,-1,-1"
            self.results.append(result)
            # Export features
            if features is not None:
                snapshot[tid] = features
            if bin_pools is not None:
                snapshot_pools[tid] = bin_pools
        self.snapshots.append(snapshot)
        self.snapshots_pools.append(snapshot_pools)

    def export_mot2d(self):
        content = "\n".join(self.results)
        with open(osp.join(self.output_dir, 'pred.txt'), 'w') as f:
            f.write(content)
        print("Export mot2d result to '{}'".format(osp.join(self.output_dir, 'pred.txt')))

    def export_snapshots(self):
        with open(osp.join(self.output_dir, 'snapshots.pkl'), 'wb') as f:
            pickle.dump(self.snapshots, f)
        print("Export features result to '{}'".format(osp.join(self.output_dir, 'snapshots.pkl')))

    def export_snapshots_pools(self):
        with open(osp.join(self.output_dir, 'snapshots_pools.pkl'), 'wb') as f:
            pickle.dump(self.snapshots_pools, f)
        print("Export feature pools result to '{}'".format(osp.join(self.output_dir, 'snapshots_pools.pkl')))
