"""This file contains code used in "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division
from thinkbayes2 import Suite
import math
import numpy
import sys

import matplotlib
import matplotlib.pyplot as pyplot

import thinkbayes2
import thinkplot

class LDR(object):
    """Represents the distribution of probability of LDR to break up"""

    def __init__(self, hypo1, num_meet, hypo2, textcall):
        
        self.meetinghypo= hypo1
        self.textcallhypo = hypo2

        self.nummeet = num_meet
        self.textcall = textcall
    
    def GetJoint(self):

        A= Meeting(self.meetinghypo)
        B= TextCall(self.textcallhypo)

        PlotJointDist(A,B)

def PlotJointDist(pmf1, pmf2, thresh=0.8):
    """
    pmf1, pmf2: posterior distributions
    thresh: lower bound of the range to be plotted
    """
    pmf = thinkbayes2.MakeJoint(pmf1, pmf2)

    thinkplot.Figure(figsize=(6, 6))    
    thinkplot.Contour(pmf, contour=False, pcolor=True)

    thinkplot.Plot([thresh, 1.0], [thresh, 1.0],
                color='gray', alpha=0.2, linewidth=4)

    thinkplot.Save(root='Joint between meeting and calling',
                   xlabel='Meeting', 
                   ylabel='Calling',
                   axis=[thresh, 1.0, thresh, 1.0],
                   formats=['pdf', 'eps'])

class Meeting(Suite):
  
    def Likelihood(self, data, hypo):
        lam = hypo / 30
        k = data
        like = thinkbayes2.EvalPoissonPmf(k, lam)
        return like


    def Prob_MeetUp(self, rem_time, meetups):
        """Computes the likelihood of breaking up"""
        
        metapmf = thinkbayes2.Pmf()
        for lam, prob in self.Items():
            lt = lam*rem_time / 30
            pred = thinkbayes2.MakePoissonPmf(lt, 5)
            metapmf[pred] = prob
        
        mix = thinkbayes2.MakeMixture(metapmf)
        mix += meetups

        thinkplot.Hist(mix)
        thinkplot.Show()

class TextCall(Suite):
   
    def Likelihood(self, data, hypo):
        lam = hypo / 30
        k = data
        like = thinkbayes2.EvalPoissonPmf(k, lam)
        return like

    def Prob_TextCall(self, rem_time, textcall):

        metapmf = thinkbayes2.Pmf()
        for lam, prob in self.Items():
            lt = lam*rem_time / 30
            pred= thinkbayes2.MakePoissonPmf(lt, 30)
            metapmf[pred] = prob
        
        mix = thinkbayes2.MakeMixture(metapmf)
        mix += textcall

        thinkplot.Hist(mix)
        thinkplot.Show()


def main(script):
    #hypos = numpy.linspace(0,6,99)
    hypos =  numpy.linspace(0,30,150)
    couple = Meeting(hypos)

    mean_meetup = 1.5 /2
    mean_meetperiod = 30/mean_meetup

    couple.Update(mean_meetperiod)
    thinkplot.Pdf(couple, label= 'prior')
    print('prior mean', couple.Mean())

    couple.Update(2)
    thinkplot.Pdf(couple,label= 'posterior1')
    couple.Update(4)
    thinkplot.Pdf(couple, label= 'posterior2')
    thinkplot.show()

    couple.Prob_MeetUp(30-16, 2)

    hypos2 = numpy.linspace(0,30,150)
    couple2 = TextCall(hypos2)

    mean_textcall = 12/2
    mean_textcall_period = 30/mean_textcall

    couple2.Update(mean_textcall_period)
    thinkplot.Pdf(couple2, label= 'prior')
    print('prior mean', couple2.Mean())

    couple2.Update(3)
    thinkplot.Cdf(couple2,label= 'posterior1')
    couple2.Update(7)
    thinkplot.Cdf(couple2, label= 'posterior2')
    couple2.Update(8)
    thinkplot.Cdf(couple2, label= 'posterior3')
    couple2.Update(13)
    thinkplot.Cdf(couple2, label= 'posterior3')
    couple2.Update(16)
    thinkplot.Cdf(couple2, label= 'posterior3')
    thinkplot.show()

    couple2.Prob_TextCall(30-16, 5)
   
    Pair = LDR(hypos, mean_meetperiod, hypos2, mean_textcall_period)
    Pair.GetJoint()
if __name__ == '__main__':
    main(*sys.argv)
