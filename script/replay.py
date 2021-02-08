import os
import os.path as osp
import sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
import time
import argparse

import cv2
import numpy as np
from PIL import Image
import pickle

from multimedia.container.msv import MultiStreamVideo
from utils.display import draw_text, draw_bbox, draw_bodypose25, draw_mask, get_color_mask, get_color


def main(args):
    # Load video
    video = MultiStreamVideo(args['video'])

    # Load vidoe result
    result = {}
    with open(args['result'], 'r') as f:
        lines = [ line for line in f.read().split('\n') if len(line) > 0 ]
        for line in lines:
            fields = [ f for f in line.split(",") if len(f) > 0 ]
            fid = int(fields[0])
            tid = fields[1]
            bbox = np.array([ float(v) for v in fields[2:2+4] ])
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            if fid not in result:
                result[fid] = []

            result[fid].append((tid, bbox))

    # Create display window
    color_winname = "MSVPlayer-RGB"
    cv2.namedWindow(color_winname, cv2.WINDOW_GUI_EXPANDED)
    if video.streams['depth'] is not None:
        depth_winname = "MSVPlayer-Depth"
        cv2.namedWindow(depth_winname, cv2.WINDOW_GUI_EXPANDED)

    # Display video
    pause = False
    verbose = False
    fid = 1
    while True:
        start_time = time.time()
        frames = video.read()
        if frames['video'] is None:
            break
        video_frame = frames['video']
        depth_frame = frames['depth']
        bodyposes = frames['bodyposes'] if frames['bodyposes'] is not None else []
        pred = result[fid] if fid in result else []

        seen = []
        for tid, bbox in pred:
            if tid in seen:
                pause = True
            seen.append(tid)
            color = get_color(tid)
            draw_bbox(video_frame, bbox, color=color, thickness=5)
            draw_text(video_frame, f"ID:{tid}",
                    position=(int(bbox[0]), int(bbox[1])),
                    fgcolor=(255, 255, 255),
                    bgcolor=color,
                    fontScale=3)
            draw_text(video_frame, f"FID:{fid}",
                    position=(0, 0),
                    fgcolor=(255, 255, 255),
                    bgcolor=(0, 0, 255),
                    fontScale=3)
            if video.streams['depth'] is not None:
                draw_bbox(depth_frame, bbox, color=color, thickness=5)
                draw_text(depth_frame, f"ID:{tid}",
                        position=(int(bbox[0]), int(bbox[1])),
                        fgcolor=(255, 255, 255),
                        bgcolor=color,
                        fontScale=3)
                draw_text(video_frame, f"FID:{fid}",
                        position=(0, 0),
                        fgcolor=(255, 255, 255),
                        bgcolor=(0, 0, 255),
                        fontScale=3)

        if verbose:
            for pose in bodyposes:
                # Plot bbox on video frame
                bbox = pose['bbox']
                draw_bbox(video_frame, bbox, thickness=5)
                # Plot mask on video frame
                mask = pose['mask']
                mask = get_color_mask(mask)
                draw_mask(video_frame, bbox, mask)
                # Plot keypoints on video frame
                keypoints = pose['keypoints']
                draw_bodypose25(video_frame, keypoints=keypoints, thickness=5)

                if video.streams['depth'] is not None:
                    draw_bbox(depth_frame, bbox, thickness=5)
                    draw_mask(depth_frame, bbox, mask)
                    draw_bodypose25(depth_frame, keypoints=keypoints, thickness=5)

        fid += 1
        # Visualize frame
        while True:
            cv2.imshow(color_winname, video_frame)
            if video.streams['depth'] is not None:
                cv2.imshow(depth_winname, depth_frame)

            elapsed_time = time.time() - start_time
            wait_time = int(1000*((1/video.fps)-elapsed_time))
            wait_time = wait_time if wait_time >= 0 else 1
            key = cv2.waitKey(wait_time) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                verbose = not verbose
            elif key == 32:
                pause = not pause

            if not pause:
                break

        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    video.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="msv video to play")
    parser.add_argument("--result", required=True, help="mot2d video result")

    args = vars(parser.parse_args())
    main(args)
