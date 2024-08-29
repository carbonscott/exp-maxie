#!/usr/bin/env python
# -*- coding: utf-8 -*-
from maxie.datasets.psana_utils import PsanaImg
from maxie.utils.data import split_list_into_chunk
import os
import yaml
import tqdm
import hydra
from omegaconf import DictConfig
import ray
import logging
logging.basicConfig(level=logging.DEBUG)

@ray.remote
def init_psana_check(exp, run, access_mode, detector_name):
    try:
        PsanaImg(exp, run, access_mode, detector_name)
        return True
    except Exception as e:
        print(f"Failed to initialize PsanaImg: {e}!!!")
        return False

@ray.remote
def process_batch(exp, run, access_mode, detector_name, events):
    psana_img = PsanaImg(exp, run, access_mode, detector_name)
    valid_events = [event for event in tqdm.tqdm(events) if psana_img.get(event, None, 'raw') is not None]
    return valid_events

def get_psana_events(exp, run, access_mode, detector_name, num_tasks=None, max_num_events=None, filename_postfix=None):
    psana_img = PsanaImg(exp, run, access_mode, detector_name)
    num_events = len(psana_img)
    if max_num_events is not None:
        num_events = max(min(max_num_events, num_events), 0)

    # If num_tasks is not specified, use the total number of CPUs available in the Ray cluster
    if num_tasks is None:
        num_tasks = ray.cluster_resources()['CPU']

    # Submit tasks to Ray
    batch_events = split_list_into_chunk(range(num_events), num_tasks)
    results = [process_batch.remote(exp, run, access_mode, detector_name, batch) for batch in batch_events]

    # Collect results
    results = ray.get(results)
    valid_events = [event for batch in results for event in batch]
    output = {
        "exp"          : exp,
        "run"          : run,
        "detector_name": detector_name,
        "events"       : valid_events,
        "num_events"   : num_events,
    }
    dir_output = "outputs"
    basename_output = f"{exp}_r{run:04d}"
    if filename_postfix is not None: basename_output += filename_postfix
    file_output = f"{basename_output}.yaml"
    path_output = os.path.join(dir_output, file_output)
    os.makedirs(dir_output, exist_ok=True)
    yaml_data = yaml.dump(output)
    with open(path_output, 'w') as file:
        file.write(yaml_data)

def run_psana(exp, run, access_mode, detector_name, num_tasks=None, max_num_events=None, filename_postfix=None):
    try:
        get_psana_events(exp, run, access_mode, detector_name, num_tasks=num_tasks, max_num_events=max_num_events, filename_postfix=filename_postfix)
    except Exception as e:
        print(f"Caught an exception: {e}!!!")

@hydra.main(config_path="hydra_config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    exp            = cfg.exp
    run            = cfg.run
    access_mode    = cfg.access_mode
    detector_name  = cfg.detector_name
    num_tasks      = cfg.num_tasks
    max_num_events = cfg.max_num_events
    postfix        = cfg.postfix

    # Initialize Ray
    ray.init(address='auto')

    # Run the initialization check as a Ray task
    init_result = ray.get(init_psana_check.remote(exp, run, access_mode, detector_name))
    if not init_result:
        print("Psana initialization failed!")
    else:
        print("Psana is launchable...")
        run_psana(exp, run, access_mode, detector_name, num_tasks, max_num_events, postfix)

    # Shut down Ray
    ray.shutdown()

if __name__ == "__main__":
    main()
