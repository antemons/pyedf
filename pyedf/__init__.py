#! /usr/bin/python

print "# Welcome to the 'European Data Format'"

import recording.channeltypes as channeltypes
from recording import recording, read_md5
from derivation import derivation, montage
from score import event, Event, state, State, score, Score, mystrtime, interval2state

