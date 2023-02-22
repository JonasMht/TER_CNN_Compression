#!/bin/bash

cat $(find . -name 'dice*.png' -print | sort -V) | ffmpeg -y -framerate 30 -i - -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output.gif

# Get path readlink -f file.ext
