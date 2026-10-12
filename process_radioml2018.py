#!/usr/bin/env python3
"""Convert RadioML 2018.01A (Kaggle: pinxau1000/radioml2018) into the
modulation-classifier format.

Thin wrapper around ``modulation_classifier.radioml2018``; see that module for
details on the class-order fix and the streaming conversion. Examples::

    # Defaults: 512 frames per (class, SNR), SNR >= 0 dB -> ~82k frames
    python process_radioml2018.py

    # Preview the selection without writing anything
    python process_radioml2018.py --dry-run

    # Larger subset
    python process_radioml2018.py --frames-per-snr 1024 --min-snr -10

    # Explicit paths
    python process_radioml2018.py --input /Volumes/ext/radioml/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \
        --classes-json /Volumes/ext/radioml/classes-fixed.json --output data
"""

import sys

from modulation_classifier.radioml2018 import main

if __name__ == "__main__":
    sys.exit(main())
