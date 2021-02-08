import os
import os.path as osp
import argparse

import motmetrics as mm

def main(args):
    gts = sorted([ osp.join(args['gt'], f) for f in os.listdir(args['gt']) ])
    preds = sorted([ osp.join(args['pred'], f) for f in os.listdir(args['pred']) ])

    accs = []
    for gt, pred in zip(gts, preds):
        gt_file = osp.join(gt, 'gt.txt')
        pred_file = osp.join(pred, 'pred.txt')

        df_gt = mm.io.loadtxt(gt_file)
        df_pred = mm.io.loadtxt(pred_file)

        acc = mm.utils.compare_to_groundtruth(df_gt, df_pred, 'iou', distth=0.5)
        accs.append(acc)

    names = [ osp.basename(f) for f in gts ]
    mh = mm.metrics.create()
    summary = mh.compute_many(accs, names=names,
                            generate_overall=True,
                            metrics=mm.metrics.motchallenge_metrics)
    print(mm.io.render_summary(summary,
                            namemap=mm.io.motchallenge_metric_names,
                            formatters=mh.formatters))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="mot2d ground truth file")
    parser.add_argument("--pred", required=True, help="mot2d prediction file")
    args = vars(parser.parse_args())
    main(args)
