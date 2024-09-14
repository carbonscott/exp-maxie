import argparse
from jinja2 import Environment, BaseLoader

def generate_bsub_script(exp, detector, runs):
    template_str = """#!/bin/bash
#BSUB -o lsf/%J.log
#BSUB -e lsf/%J.err
#BSUB -q batch
#BSUB -W 2:00
#BSUB -P LRN044
#BSUB -J p2z-{{ exp }}
#BSUB -nnodes 1

OMP_NUM_THREADS=1 jsrun -n1 -a42 -c42 -g0 python psana_to_zarr.py {{ exp }} {{ detector }} --partition-size 512 --output-dir output --run {% for run in runs %}{{ run }} {% endfor %}

jskill all
"""

    env = Environment(loader=BaseLoader)
    template = env.from_string(template_str)
    script_content = template.render(exp=exp, detector=detector, runs=runs)

    filename = f"psana_to_zarr.{exp}.bsub"
    with open(filename, 'w') as f:
        f.write(script_content)
    print(f"Bash script '{filename}' has been generated.")

def main():
    parser = argparse.ArgumentParser(description="Generate a bash script for psana_to_zarr processing.")
    parser.add_argument("exp", help="Experiment name")
    parser.add_argument("detector", help="Detector name")
    parser.add_argument("--run", dest="runs", nargs="+", type=int, required=True, help="Run numbers to process")

    args = parser.parse_args()

    generate_bsub_script(args.exp, args.detector, args.runs)

if __name__ == "__main__":
    main()
