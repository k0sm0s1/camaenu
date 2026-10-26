#!/bin/bash
# usage: scan.sh outfile "ENV1=.. ENV2=.." "ENV..." ...   (runs 2 at a time)
out=$1; shift
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa STACK_RENDER_EVERY=25
: > $out
printf '%s\0' "$@" | xargs -0 -P2 -I{} bash -c 'r=$(env {} python3 tune_run.py 2>&1 | grep RESULT | sed "s/^RESULT //"); echo "{} => $r" >> '"$out"
