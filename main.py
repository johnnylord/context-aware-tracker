import sys
import os
import os.path as osp
import argparse
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from tracker import get_tracker_cls
from multimedia.container.msv import MultiStreamVideo
from utils.experiment import TrackAccumulator
from utils.display import (
        get_color, get_color_mask,
        draw_bbox, draw_text, draw_gaussian, draw_bodypose25, draw_mask
        )


def main(args):
    # Load video
    video = MultiStreamVideo(args['video'])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter("output.mp4", fourcc, video.fps, (video.width, video.height))

    # Create track accumulator
    ta = TrackAccumulator(output_dir=args['output'])

    # Create tracker
    tracker_cls = get_tracker_cls(args['tracker'])
    tracker = tracker_cls(target_video=video,
                        max_depth=args['maxdepth'],
                        n_depth_levels=args['nlevel'])
    print(tracker)

    # Show tracking result
    cv2.namedWindow('video', cv2.WINDOW_GUI_EXPANDED)
    if args['tracker'] == 'CAT':
        cv2.namedWindow('depth', cv2.WINDOW_GUI_EXPANDED)
        if args['birdeye']:
            cv2.namedWindow('birdeye', cv2.WINDOW_GUI_EXPANDED)

    pause = False
    for fid, (frames, tracks) in enumerate(tracker):
        ta.accumulate(fid, tracks)
        video_frame = frames['video']
        depth_frame = frames['depth']
        # End of video
        if video_frame is None:
            break
        # Draw keypoints
        if len(frames['bodyposes']) > 0:
            bodyposes = frames['bodyposes']
            for pose in bodyposes:
                bbox = pose['bbox']
                mask = get_color_mask(pose['mask'])
                if args['verbose']:
                    draw_bodypose25(video_frame, pose['keypoints'])
                    draw_mask(video_frame, bbox, mask)
                if args['tracker'] == 'CAT':
                    bbox = pose['bbox']
                    mask = get_color_mask(pose['mask'])
                    if args['verbose']:
                        draw_bodypose25(depth_frame, pose['keypoints'])
                        draw_mask(depth_frame, bbox, mask)
        # Draw track on the video
        for t in tracks:
            tid, bbox = t['id'], t['bbox']
            mean, covar = t['mean'], t['covar']
            cx, cy = int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)
            # Draw on video frame
            color = get_color(tid)
            text = f"ID:{tid}"
            if 'status' in t and t['status'] != 'Stand':
                text += f"({t['status']})"
                color = (0, 0, 255)
            draw_bbox(video_frame, bbox, color=color)
            draw_text(video_frame, text, tuple(bbox[:2]),
                        fgcolor=(255, 255, 255),
                        bgcolor=color,
                        fontScale=2)
            draw_gaussian(video_frame, mean[:2], covar[:2, :2], color=color)
            # Draw on depth frame
            if args['tracker'] == 'CAT':
                draw_bbox(depth_frame, bbox, color=color)
                draw_text(depth_frame, text, tuple(bbox[:2]),
                        fgcolor=(255, 255, 255),
                        bgcolor=color,
                        fontScale=2)
                draw_gaussian(depth_frame, mean[:2], covar[:2, :2], color=color)
        # Draw processing time
        if args['verbose']:
            text = f"Frame: {fid}"
            text += f"\nSpeed: {tracker.fps} (fps)"
            draw_text(video_frame, text, (0, 0),
                    fgcolor=(255, 255, 255),
                    bgcolor=(0, 0, 255),
                    fontScale=3, margin=10)
            if args['tracker'] == 'CAT':
                draw_text(depth_frame, text, (0, 0),
                        fgcolor=(255, 255, 255),
                        bgcolor=(0, 0, 255),
                        fontScale=3, margin=10)
        # Display on screen
        writer.write(video_frame)
        cv2.imshow('video', video_frame)
        if args['tracker'] == 'CAT':
            cv2.imshow('depth', depth_frame)
            if args['birdeye']:
                cv2.imshow('birdeye', tracker.bird_view)
        # Waiting time to approximate precise fps
        wait_time = int((1/video.fps-1/tracker.fps)*1000)
        wait_time = wait_time if wait_time > 0 else 1
        while True:
            key = cv2.waitKey(wait_time) & 0xff
            # Keyboard handling
            if key == ord('q'):
                break
            elif key == 32:
                pause = not pause
            # Pause video
            if not pause:
                break

        if key == ord('q'):
            break

    ta.export_mot2d()
    ta.export_snapshots()
    ta.export_snapshots_pools()
    # Close resources
    cv2.destroyAllWindows()
    video.close()
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=str, help="msv video to process")
    parser.add_argument("--tracker", default="CAT", type=str, help="tracker you want to use")
    parser.add_argument("--maxdepth", default=4, type=int, help="maximum depth in space")
    parser.add_argument("--nlevel", default=5, type=int, help="discrete space between depth range")
    parser.add_argument("--birdeye", action='store_true', help="show birdeye view")
    parser.add_argument("--verbose", action='store_true', help="show metadata")
    parser.add_argument("--output", default="result", help="output directory")
    args = vars(parser.parse_args())
    main(args)
