import thinkbayes2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import thinkplot


beta = [2,1]
"beta= parameters for meetup, calls"


class LDR(thinkbayes2.Suite, thinkbayes2.Joint):
	"class that has suite of LDR couples"
	
	def Likelihood(self, data, hypo):

		pass



	def Update(self, data):

		pass


def dataframe_to_lists(dataframe,i):
    """
    i=0: meet, i=1: call
    Convert from a pandas dataframe to two lists, dates = [date1, date2, date3... ] (ns), records = [12.1, 12.3] (mph)
    """
    data = [row[i].value for i, row in enumerate(dataframe.values)]
    records = [row[2] for i, row in enumerate(dataframe.values)]
    return dates, records

def MeetPmf(csv):
	Meetdataframe = pd.read_csv(ldr_csv.csv, skiprows=[1])

def computing_likelihood_beta(beta, data):
	"data[0] = y, data[1] = x1, data[2] = x2"
	y= data[0]
	x1= data[1]
	x2= data[2]
	log_o = beta[0]*x1+ beta[1]* x2
	o = np.exp(log_o)
	p = o/(1+o)

	likes = y * p + (1-y) * (1-p)
	like = np.prod(likes)

	return like