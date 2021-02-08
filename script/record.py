import os
import os.path as osp
import traceback
import argparse
import cv2
import numpy as np
import pyrealsense2 as rs

SETTINGS = {
    'l515': {
        'scale': 0.0002500000118743628,
        'depth': {
            'fps': 30,
            'resolution': (1024, 768),
        },
        'color': {
            'fps': 30,
            'resolution': (1920, 1080),
        }
    },
    'd455': {
        'scale': 0.0010000000474974513,
        'depth': {
            'fps': 30,
            'resolution': (1280, 720),
        },
        'color': {
            'fps': 30,
            'resolution': (1280, 720),
        }
    },
}

def main(args):
    setting = SETTINGS[args['model']]

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    # different resolutions of color and depth streams
    config = rs.config()

    depth_fps = setting['depth']['fps']
    depth_width, depth_height = setting['depth']['resolution']
    config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, depth_fps)

    color_fps = setting['color']['fps']
    color_width, color_height = setting['color']['resolution']
    config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, color_fps)

    # Start streaming
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print("Depth Scale:", depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Create colorzier
    colorizer = rs.colorizer(2)

    # Create display window
    cv2.namedWindow('RGB', cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow('Depth', cv2.WINDOW_GUI_EXPANDED)

    # Create video writer
    if args['export']:
        if not osp.exists(args['output']):
            os.makedirs(args['output'])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = osp.join(args['output'], 'video.mp4')
        depth_path = osp.join(args['output'], 'depth.mp4')
        video_writer = cv2.VideoWriter(video_path, fourcc, color_fps, (color_width, color_height))
        depth_writer = cv2.VideoWriter(depth_path, fourcc, color_fps, (color_width, color_height))

    # Streaming loop
    depth_frame_buffers = []
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if (
                not depth_frame
                or not color_frame
            ):
                continue

            # Convert frame to numpy data type
            depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Render images
            cv2.imshow('RGB', color_image)
            cv2.imshow('Depth', depth_image)

            if args['export']:
                video_writer.write(color_image)
                depth_writer.write(depth_image)

            # Press esc or 'q' to close the image window
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break

    except Exception as e:
        traceback.print_exc()

    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="camera model")
    parser.add_argument("--export", action='store_true', help="export recorded video")
    parser.add_argument("--output", type=str, default="output", help="output directory")

    args = vars(parser.parse_args())
    main(args)
