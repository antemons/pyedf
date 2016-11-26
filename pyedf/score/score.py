#! /usr/bin/python

from __future__ import with_statement

import logging
from .state import State
import numpy as np
import os
import re



class Score(object):

    logger = logging.getLogger(name='Score')

    commentSymbol = '#'    # used for comments in state.annot
    lineSeparator = ','    # used as separators in the line
    states_dict = dict()

    def __init__(self, filename=None, states=[], verbose=0):
        self.logger.debug("__init__(filename={}, num_states={})".format(filename, states))
        self.verbose = verbose
        self.states  = states
        self.filename = filename

        if not self.filename is None:

            if not os.path.exists(self.filename):
                print("Score file %s does not exist." % (filename))
                raise AttributeError

            self.states = self.load(filename)
            if self.verbose > 0:  print("score: score file '%s' found." % (filename))
            if self.verbose == 2: print("score: the states", self.states)


    def interpret_states(self):
        pass


    def isComment(self, line):

        line.strip(' ')

        if line[0] == self.commentSymbol:    # if line starts with the commentSymbol, it is a comment ..
            return True            #                     .. don't process it.

        else:
            return False    # else: split the line at separators.


    def load(self, filename):
        self.logger.debug("load(filename='{}')".format(filename))
        states = []
        with open(filename, 'r') as score_file:
            for line in score_file:
                try:
                    if self.isComment(line):
                        continue
    
                    line = line.strip('\n').strip('\r').strip(' ')
                    x = line.split(self.lineSeparator)
    
                    if len(x) > 0:                    # for example 1
                        start = x[0].strip(' ')
    
                    if len(x) == 1:
                        annot     = ''
                        duration = ''
    
                    if len(x) == 2:
                        annot     = x[1]
                        duration = ''
    
                    elif len(x) > 2:                # for example 3.
                        duration = x[1].strip(' ')
                        annot    = x[2]
    
                    if duration == '':
                        duration = '-1'
        
                    states.append( State(start=start, duration=duration, annot=annot)  )
    
                except Exception as e:
                    self.logger.debug("# line not readable: {}\n{}".format(line, e))
        return states


    def save(self, filename):

        print("# opening", filename, "to write ...")
        score_file = open(filename, 'w')
        string = '# start, duration, annotation\n'+self.__str__()
        score_file.write(string + '\n')
        score_file.close()


    def append(self, new_state=None, start=None, duration=None, annot=None):

        if new_state == None:
            new_state = State(start=start, duration=duration, annot=annot)

        self.states.append(new_state)


    def __str__(self):

        return '\n'.join([str(state) for state in self.states])


    def select_by_function(self, function, **kwargs):

        selection = []
        for state in self.states:
            
            if function(state, **kwargs):
                selection.append(state)

        score_select = object.__new__(type(self))
        score_select.__init__(states=selection)
        return score_select


    def intersect(self, other_score):

        intersection = []

        for state in self.states:

            section_j = state.intersect(other_score.states)
            intersection.extend( section_j )
    
        return type(self)(states=intersection)


    def duration(self, annot=None):

        duration = 0.0
        if annot == None:
            duration = np.sum([state.duration for state in self.states])

        else:
            for state in self.states:
                if state.annot == annot:
                    duration += state.duration

        return duration


    def count(self, annot=None):

        if annot == None:
            count = len(self.states)

        else:
            count = 0

            for state in self.states:

                if state.annot == annot:
                    count += 1

        return count



score = Score




if __name__ == "__main__":

    score_filename = '../../example/sample.csv'

    testscore = score(filename=score_filename)

    print(testscore)
    



