import os
import os.path as osp
import sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import time
import tempfile
import argparse

import io
import cv2
from scipy.optimize import linear_sum_assignment
import numpy as np
from PIL import Image
import pickle

import grpc
import message.person_reid_pb2 as person_reid_pb2
import message.object_detection_pb2 as object_detection_pb2
import message.object_segmentation_pb2 as object_segmentation_pb2
import message.pose_estimation_pb2 as pose_estimation_pb2
import service.person_reid_pb2_grpc as person_reid_pb2_grpc
import service.object_detection_pb2_grpc as object_detection_pb2_grpc
import service.object_segmentation_pb2_grpc as object_segmentation_pb2_grpc
import service.pose_estimation_pb2_grpc as pose_estimation_pb2_grpc

from multimedia.container.msv import MultiStreamVideo
from utils.convert import pil_to_bytes, get_angle_2d
from utils.transform import scale_bboxes_coord, scale_keypoints_coord, scale_masks_coord
from utils.display import draw_text, draw_bbox, draw_bodypose25, draw_mask, get_color_mask
from utils.cost import compute_iou_dist


REID_SERVER_IP = "140.112.18.214"
REID_SERVER_PORT = 50001
REID_SIZE = (128, 256)

OBJECT_DETECTION_SERVER_IP = "140.112.18.214"
OBJECT_DETECTION_SERVER_PORT = 50002

POSE_ESTIMATION_SERVER_IP = "140.112.18.214"
POSE_ESTIMATION_SERVER_PORT = 50003

OBJECT_SEGMENTATION_SERVER_IP = "140.112.18.214"
OBJECT_SEGMENTATION_SERVER_PORT = 50004

FRAME_SIZE = (512, 512)

def extract_bboxes(video_frame, segmentation_service):
    frame = cv2.resize(video_frame, FRAME_SIZE)
    img = Image.fromarray(frame)

    # Perform object segmentation
    request = object_segmentation_pb2.SegmentRequest()
    request.img.payload = pil_to_bytes(img)
    response = segmentation_service.SegmentObjects(request)

    # Filter out person bbox & mask
    bboxes, masks = [], []
    for bbox in response.bboxes:
        if bbox.label != 'person':
            continue
        # BBOX
        xmin, ymin = bbox.xmin, bbox.ymin
        xmax, ymax = bbox.xmax, bbox.ymax
        conf = bbox.conf
        # MASK
        buf = io.BytesIO(bbox.mask.payload)
        mask = Image.open(buf)
        mask = np.array(mask, dtype=np.uint8)
        # RECORD
        bboxes.append([xmin, ymin, xmax, ymax, conf])
        masks.append(mask)

    # Rescale coordinate to target resolution
    if len(bboxes) > 0:
        bboxes = scale_bboxes_coord(np.array(bboxes),
                                    old_resolution=frame.shape[:2],
                                    new_resolution=video_frame.shape[:2])
        new_masks = []
        for bbox, mask in zip(bboxes, masks):
            width = int(bbox[2])-int(bbox[0])
            height = int(bbox[3])-int(bbox[1])
            mask = mask.astype(np.uint8)
            mask = cv2.resize(mask, (width, height))
            new_masks.append(mask)
        masks = new_masks

    return bboxes, masks

def extract_keypoints(video_frame, estimation_service):
    img = Image.fromarray(video_frame)

    # Perform object detection
    request = pose_estimation_pb2.EstimateRequest()
    request.img.payload = pil_to_bytes(img)
    response = estimation_service.EstimatePoses(request)

    # Extract keypoints
    all_keypoints = []
    for pose in response.poseKeypoints:
        points = np.array([ (p.x, p.y, p.conf) for p in pose.points ])
        all_keypoints.append(points)
    all_keypoints = np.array(all_keypoints)

    return all_keypoints

def extract_features(video_frame, bboxes, reid_service):
    # Perform person reidentification
    reid_requests = []
    for bbox in bboxes:
        xmin, ymin = int(bbox[0]), int(bbox[1])
        xmax, ymax = int(bbox[2]), int(bbox[3])
        crop_img = video_frame[ymin:ymax, xmin:xmax, :]
        crop_img = cv2.resize(crop_img, REID_SIZE)
        crop_img = Image.fromarray(crop_img)
        request = person_reid_pb2.FeatureRequest()
        request.img.payload = pil_to_bytes(crop_img)
        reid_requests.append(request)
    reid_responses = reid_service.GetFeatures(iter(reid_requests))

    # Convert features type
    features = np.array([np.array(response.vector)
                        for response in reid_responses ])
    return features

def aggregate_information(bboxes, masks, all_keypoints, features):
    # Keypoints to bounding boxes
    kbboxes = []
    for keypoints in all_keypoints:
        valids = keypoints[(keypoints[:, 0]*keypoints[:, 1])>0, :]
        xmin = np.min(valids[:, 0])
        xmax = np.max(valids[:, 0])
        ymin = np.min(valids[:, 1])
        ymax = np.max(valids[:, 1])
        kbboxes.append([xmin, ymin, xmax, ymax])
    kbboxes = np.array(kbboxes)

    # Group bboxes with kbboxes
    bodyposes = []
    if len(bboxes) > 0 and len(kbboxes) > 0:
        iou_matrix = compute_iou_dist(bboxes[:, :4], kbboxes)
        bindices, kindices = linear_sum_assignment(iou_matrix)
        for bidx, kidx in zip(bindices, kindices):
            mask = masks[bidx]
            bbox = bboxes[bidx]
            feature = features[bidx]
            keypoints = all_keypoints[kidx]
            pose = { 'mask': mask, 'bbox': bbox, 'feature': feature, 'keypoints': keypoints }
            bodyposes.append(pose)

    return bodyposes

def main(args):
    # Load video
    video = MultiStreamVideo(args['path'])
    process_streams = dict([ (stream_name, [])
                            for stream_name, config in MultiStreamVideo.STREAMS.items()
                            if config['importable'] ])

    # Create display window
    color_winname = "MSVPlayer-RGB"
    cv2.namedWindow(color_winname, cv2.WINDOW_GUI_EXPANDED)
    if video.streams['depth'] is not None:
        depth_winname = "MSVPlayer-Depth"
        cv2.namedWindow(depth_winname, cv2.WINDOW_GUI_EXPANDED)

    # Connect to serivce provider
    with \
        grpc.insecure_channel(f"{REID_SERVER_IP}:{REID_SERVER_PORT}") as reid_channel, \
        grpc.insecure_channel(f"{OBJECT_DETECTION_SERVER_IP}:{OBJECT_DETECTION_SERVER_PORT}") as detection_channel, \
        grpc.insecure_channel(f"{POSE_ESTIMATION_SERVER_IP}:{POSE_ESTIMATION_SERVER_PORT}") as estimation_channel, \
        grpc.insecure_channel(f"{OBJECT_SEGMENTATION_SERVER_IP}:{OBJECT_SEGMENTATION_SERVER_PORT}") as segmentation_channel \
    :
        reid_service = person_reid_pb2_grpc.PersonReIDStub(reid_channel)
        detection_service = object_detection_pb2_grpc.DetectionStub(detection_channel)
        estimation_service = pose_estimation_pb2_grpc.EstimationStub(estimation_channel)
        segmentation_service = object_segmentation_pb2_grpc.SegmentationStub(segmentation_channel)

        # Display video
        pause = False
        while True:
            start_time = time.time()
            frames = video.read()
            if frames['video'] is None:
                break
            video_frame = frames['video']
            depth_frame = frames['depth']

            # Online processing video
            if args['process']:

                # Extract bounding boxes & masks
                bboxes, masks = extract_bboxes(video_frame, segmentation_service)

                # Extract keypoints
                all_keypoints = extract_keypoints(video_frame, estimation_service)

                # Extract features
                features = extract_features(video_frame, bboxes, reid_service)

                # Post processing
                bodyposes = aggregate_information(bboxes, masks, all_keypoints, features)

                # Record meta result
                if args['process']:
                    process_streams['bodyposes'].append(bodyposes)
            else:
                bodyposes = frames['bodyposes'] if frames['bodyposes'] is not None else []

            for pose in bodyposes:
                # Plot bbox on video frame
                bbox = pose['bbox']
                area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                if area < 68100 or area > 543000:
                    continue
                if args['verbose']:
                    draw_bbox(video_frame, bbox, thickness=5)
                    # Plot mask on video frame
                    mask = pose['mask']
                    mask = get_color_mask(mask)
                    draw_mask(video_frame, bbox, mask)
                    # Plot keypoints on video frame
                    keypoints = pose['keypoints']
                    draw_bodypose25(video_frame, keypoints=keypoints, thickness=5)
                    # Plot pose orientation
                    angle, conf = get_angle_2d(keypoints)
                    text = "Angle:{:.2f}".format(angle)
                    draw_text(video_frame, text,
                            position=(int(bbox[0]), int(bbox[1])),
                            fontScale=1.5,
                            fgcolor=(255, 255, 255))
                if depth_frame is not None:
                    if args['verbose']:
                        draw_bbox(depth_frame, bbox, thickness=5)
                        draw_mask(depth_frame, bbox, mask)
                        draw_bodypose25(depth_frame, keypoints=keypoints, thickness=5)
                        draw_text(depth_frame, text,
                                position=(int(bbox[0]), int(bbox[1])),
                                fontScale=1.5,
                                fgcolor=(255, 255, 255))

            # Visualize frame
            cv2.imshow(color_winname, video_frame)
            if depth_frame is not None:
                cv2.imshow(depth_winname, depth_frame)

            elapsed_time = time.time() - start_time
            wait_time = int(1000*((1/video.fps)-elapsed_time))
            wait_time = wait_time if wait_time >= 0 else 1
            key = cv2.waitKey(wait_time) & 0xFF
            # Keyboard handling
            if key == ord('q'):
                break
            elif key == 32:
                pause = not pause

            while pause:
                key = cv2.waitKey(wait_time) & 0xFF
                if key == ord('q'):
                    break
                elif key == 32:
                    pause = not pause

            if key == ord('q'):
                break

    # Save video stream
    if args['process'] and args['export']:
        for k, v in process_streams.items():
            with tempfile.NamedTemporaryFile(delete=False) as f:
                pickle.dump(v, f)
                process_streams[k] = f.name

        video.import_stream(process_streams)
        video.export()

    cv2.destroyAllWindows()
    video.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="msv video to play")
    parser.add_argument("--process", action='store_true', help="online process stream")
    parser.add_argument("--export", action='store_true', help="export processed video streams")
    parser.add_argument("--verbose", action='store_true', help="export processed video streams")

    args = vars(parser.parse_args())
    main(args)
