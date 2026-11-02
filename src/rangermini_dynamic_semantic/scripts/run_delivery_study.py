#!/usr/bin/env python3
"""Whole study: original selectors, input controls, grounded interface follow-up."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();here=Path(__file__).resolve().parent
command=[sys.executable,str(here/'run_delivery_grounding_benchmark.py'),'--output',str(a.output)]
if a.resume:command.append('--resume')
subprocess.run(command,check=True)
subprocess.run([sys.executable,str(here/'run_grounded_delivery_variant.py'),'--root',str(a.output)],check=True)
