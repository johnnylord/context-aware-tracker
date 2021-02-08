import os
import os.path as osp
import argparse

"""
Modify REMAP global variable
 - key: remapped track ID
 - value: list of track IDs that should be remapped to the key
"""
REMAP = {
    # Example
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
}

def main(args):
    # Aggregate valid track ID
    VALID_IDs = []
    for tids in REMAP.values():
        VALID_IDs.extend(tids)
    # Remap trackd ID
    with open(args['path'], "r") as f:
        results = []
        lines = [ line for line in f.read().split("\n") if len(line) > 0 ]
        for line in lines:
            fields = line.split(",")
            fid = int(fields[0])
            tid = int(fields[1])
            bb_left = float(fields[2])
            bb_top = float(fields[3])
            bb_width = float(fields[4])
            bb_height = float(fields[5])
            # Filter out invalid ID (false positive)
            if tid not in VALID_IDs:
                continue
            # Remap track ID
            for k, v in REMAP.items():
                if tid in v:
                    tid = k
                    break
            result = f"{fid},{tid},{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},-1,-1,-1,-1"
            results.append(result)
    # Export new result file
    with open(args['path'], "w") as f:
        content = "\n".join(results)
        f.write(content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="mot2d result to be remapped")
    args = vars(parser.parse_args())
    main(args)
