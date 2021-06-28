import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.utils.transform_got10k import transform_got10k


def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='gott', sequence=None, debug=0, threads=0,
                num_gpus=8):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)




if __name__ == '__main__':
    run_tracker("stark_s", "baseline_got10k_only",  dataset_name="got10k_test", num_gpus=1)
    transform_got10k("stark_s", "baseline_got10k_only")