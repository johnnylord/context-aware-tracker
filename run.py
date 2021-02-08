import os
import os.path as osp
import argparse

from tracker import get_tracker_cls
from multimedia.container.msv import MultiStreamVideo
from utils.experiment import TrackAccumulator


def main(args):
    # tracking video to process
    video_files = [ osp.join(args['input'], f)
                    for f in os.listdir(args['input'])
                    if 'msv' in f ]

    # Process video one-by-one
    for video_file in video_files:
        # Load video
        video = MultiStreamVideo(video_file)

        # Create track accumulator
        output_dir = osp.join(args['output'], osp.basename(video_file))
        if not osp.exists(output_dir):
            os.makedirs(output_dir)

        # Create tracking result accumulator
        ta = TrackAccumulator(output_dir=output_dir)

        # Create tracker
        tracker_class = get_tracker_cls(args['tracker'])
        tracker = tracker_class(target_video=video,
                                max_depth=args['maxdepth'],
                                n_depth_levels=args['nlevel'])

        # Accumulate tracking result
        for fid, (frames, tracks) in enumerate(tracker):
            if frames['video'] is None:
                break
            ta.accumulate(fid, tracks)

        # Export result
        ta.export_mot2d()
        ta.export_snapshots()
        ta.export_snapshots_pools()
        video.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Tracking I/O
    parser.add_argument("--input", required=True, type=str, help="directory contains msv video to process")
    parser.add_argument("--output", required=True, type=str, help="directory to save mot2d tracking result for each msv video")
    # Tracker setting
    parser.add_argument("--tracker", default="SORT", type=str, help="tracker you want to use")
    parser.add_argument("--maxdepth", default=4, type=int, help="maximum depth in space")
    parser.add_argument("--nlevel", default=5, type=int, help="discrete space between depth range")

    args = vars(parser.parse_args())
    main(args)
