import os
import os.path as osp
import argparse

import cv2
import numpy as np
import pickle

def export_seqinfo(output_dir, seqinfo):
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(osp.join(output_dir, 'seqinfo.ini'), 'w') as f:
        f.write('[Sequence]\n')
        orders = [ 'name', 'imDir', 'frameRate', 'seqLength', \
                'imWidth', 'imHeight', 'imExt' ]
        for k in orders:
            v = str(seqinfo[k])
            f.write("{}={}\n".format(k, v))

def export_imgframe(output_dir, cap):
    img_dir = osp.join(output_dir, 'img1')
    if not osp.exists(img_dir):
        os.makedirs(img_dir)
    idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(osp.join(img_dir, "%06d.jpg" % idx), frame)
        idx += 1

def export_gt(output_dir, gt_path):
    gt_dir = osp.join(output_dir, 'gt')
    if not osp.exists(gt_dir):
        os.makedirs(gt_dir)

    # Format fid,tid,bb_left,bb_top,bb_width,bb_height,1,1,1
    results = {}
    with open(gt_path, 'r') as f:
        lines = [ line for line in f.read().split('\n') if len(line) > 0 ]
        for line in lines:
            fields = line.split(',')
            fid = int(fields[0])
            tid = int(fields[1])
            bb_left = int(float(fields[2]))
            bb_top = int(float(fields[3]))
            bb_width = int(float(fields[4]))
            bb_height = int(float(fields[5]))
            if tid not in results:
                results[tid] = []
            results[tid].append((fid, (bb_left, bb_top, bb_width, bb_height)))
    for k in results:
        results[k] = sorted(results[k], key=lambda p: p[0])

    with open(osp.join(gt_dir, 'gt.txt'), 'w') as f:
        tids = sorted(list(results.keys()))
        for tid in tids:
            result = results[tid]
            for pair in result:
                fid = pair[0]
                bb_left = pair[1][0]
                bb_top = pair[1][1]
                bb_width = pair[1][2]
                bb_height = pair[1][3]
                f.write(f'{fid},{tid},{bb_left},{bb_top},{bb_width},{bb_height},1,1,1\n')

def export_det(output_dir, bodyposes):
    det_dir = osp.join(output_dir, 'det')
    if not osp.exists(det_dir):
        os.makedirs(det_dir)

    with open(osp.join(det_dir, 'det.txt'), 'w') as f:
        for idx, poses in enumerate(bodyposes):
            fid = idx+1
            for pose in poses:
                bb_left = pose['bbox'][0]
                bb_top = pose['bbox'][1]
                bb_width = pose['bbox'][2] - pose['bbox'][0]
                bb_height = pose['bbox'][3] - pose['bbox'][1]
                conf = pose['bbox'][4]
                f.write(f'{fid},-1,{bb_left},{bb_top},{bb_width},{bb_height},{conf},-1,-1,-1\n')

def main(args):
    # For convenient access
    dataset_dir = args['dataset']
    groundtruth_dir = args['groundtruth']
    output_dir = args['output']
    output_video_dir = osp.join(output_dir, 'MOT17Det', 'train')
    output_label_dir = osp.join(output_dir, 'MOT17Labels', 'train')
    if not osp.exists(output_video_dir):
        os.makedirs(output_video_dir)
    if not osp.exists(output_label_dir):
        os.makedirs(output_label_dir)
    if not osp.exists(osp.join(osp.dirname(output_video_dir), 'test')):
        os.makedirs(osp.join(osp.dirname(output_video_dir), 'test'))
    if not osp.exists(osp.join(osp.dirname(output_label_dir), 'test')):
        os.makedirs(osp.join(osp.dirname(output_label_dir), 'test'))

    # Pairing
    videos_dir = sorted([ osp.join(dataset_dir, v) for v in os.listdir(dataset_dir) ])
    results_dir = sorted([ osp.join(groundtruth_dir, v) for v in os.listdir(groundtruth_dir) ])
    triplets = [(osp.join(vdir, 'video.mp4'),
                osp.join(vdir, 'bodyposes.pkl'),
                osp.join(rdir, 'gt.txt'))
                for vdir, rdir in zip(videos_dir, results_dir) ]

    # Convert dataset one-by-one
    for idx, (video_path, bodyposes_path, gt_path) in enumerate(triplets):
        name = "MOT17-%02d" % (idx+1)
        print("'{}':'{}'".format(name, osp.basename(osp.dirname(video_path))))
        target_video_dir = osp.join(output_video_dir, name)
        target_label_dir = osp.join(output_label_dir, name)

        # Process video
        # ======================================================
        cap = cv2.VideoCapture(video_path)
        seqinfo = {
            'name': name,
            'imDir': 'img1',
            'frameRate': int(cap.get(cv2.CAP_PROP_FPS)),
            'seqLength': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'imWidth': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'imHeight': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'imExt': '.jpg',
        }
        export_seqinfo(target_video_dir, seqinfo)
        export_imgframe(target_video_dir, cap)
        export_gt(target_video_dir, gt_path)

        # Process label data
        # ======================================================
        with open(bodyposes_path, 'rb') as f:
            bodyposes = pickle.load(f)

        export_seqinfo(target_label_dir+"-DPM", seqinfo)
        export_det(target_label_dir+"-DPM", bodyposes)
        export_gt(target_label_dir"-DPM", gt_path)

        # Close resource
        cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset root directory")
    parser.add_argument("--groundtruth", help="groundtruth root directory")
    parser.add_argument("--output", default="MOT", help="converted MOT root directory")
    args = vars(parser.parse_args())
    main(args)
