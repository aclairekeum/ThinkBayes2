import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import thinkbayes2
import thinkplot

LDRcsv = 'ldr.csv'
#file name for data

class LDR(thinkbayes2.Suite, thinkbayes2.Joint):

	def Likelihood(self, data, hypo):
		"""
		Calculate the likelihood of a hypo given the dataset
		data: row[0] = 1: breakup / 0: didn't breakup (boolean)
			  row[1] = how many times they meet in a monthly basis. (frequency/months)
			  row[2] = how long total they talk to each other in a weekly basis (mins)
		period of meetups or calls
		"""

		beta0, beta1, beta2, sigma = hypo #Assign the hypos to specific variables

		#For each dataset, calculate the probability from log_odds
		for i, row in enumerate(data.values): 
			meetups = row[2]
			talks = row[1]
			breakup = row[0]
			log_breakup_odds = beta0 + beta1 * meetups + beta2 * talks + sigma
			breakup_prob = logo_to_p(log_breakup_odds)

		#Get the likelihood by comparing with the result: brokeup/didn't break up
			if breakup:
				like_breakup = breakup_prob
			else:
				like_breakup = 1-breakup_prob
			return like_breakup

def makedata(filename):
	"Returns dataframe from csv file"
	df = pd.read_csv(filename,
					names=['relationship','meet','talk'],
					header=0,
					)
	return df

def logo_to_p(logo):
	"Converts log odds to probability"
	o = np.exp(logo)
	p = o / (1+o)
	# prior_odds = prior_p/(1-prior_p)
	# post_odds = prior_odds * o
	# p = post_odds/(post_odds+1)

	return p

def pmf_from_data(filename, params):
	"Returns PMF with the list of hypothesis"
	LDRdf = makedata(filename)

	b0 = params[0]
	b1 = params[1]
	b2 = params[2]

	b0est_range = 2.5
	b1est_range = 0.2
	b2est_range = 5
	b0hypos = np.linspace(b0*(1-b0est_range), b0*(1+b0est_range),20)
	b1hypos = np.linspace(b1*(1-b1est_range), b1*(1+b1est_range),20)
	b2hypos = np.linspace(b2*(1-b2est_range), b2*(1+b2est_range),20)
	sigmahypos = np.linspace(0.001,0.05,20)

	hypos = [(b0hypo,b1hypo,b2hypo,sigmahypo) for b0hypo in b0hypos for b1hypo in b1hypos for b2hypo in b2hypos for sigmahypo in sigmahypos]
	ldr_pmf = LDR(hypos)
	ldr_pmf.Update(LDRdf)

	return ldr_pmf

def modelaccuracy_with_fixedbeta(dataset,betas_mean):
	beta0 = betas_mean[0]
	beta1 = betas_mean[1]
	beta2 = betas_mean[2]
	sigma = betas_mean[3]
	correct = 0 
	incorrect = 0
	percentage = 0
	for i, row in enumerate(dataset.values): 
			meetups = row[2]
			talks = row[1]
			breakup = row[0]
			log_breakup_odds = beta0 + beta1 * meetups + beta2 * talks + sigma
			breakup_prob = logo_to_p(log_breakup_odds)

			print breakup_prob
			if breakup_prob<0.5 and breakup==0:
				correct+=1
			elif breakup_prob>0.5 and breakup==1:
				correct+=1
			else:
				incorrect+=1

	return 1.00* correct/i
	
def main(script):

	beta = [0, 0, 0]

	df = makedata(LDRcsv)
	formula = 'relationship ~ meet+talk'
	results = smf.logit(formula, data=df).fit()
	print results.summary()

	#Distributes beta values
	beta[0] = results.params['Intercept']
	beta[1] = results.params['meet']
	beta[2] = results.params['talk']

	# predict = (results.predict() >= 0.5)
	# true_pos= predict*actual
	# true_neg = (1-predict) * (1-actual)
	# acc=  (sum(true_pos) + sum(true_neg))/len(actual)
	# print acc

	#Create PMFs from dataset
	ldr_pmf = pmf_from_data(LDRcsv,beta)

	#The code below was taken from cypressf/thinkbayes2 to calculate and plot the maximum Likelihood
	maximum_likelihoods = [0, 0, 0, 0]
	for title, i in [('b0', 0), ('b1', 1), ('b2', 2),('sigma',3)]:
		marginal = ldr_pmf.Marginal(i)
		maximum_likelihoods[i] = marginal.MaximumLikelihood()
		thinkplot.Hist(marginal)
		plt.title("PMF for " + title)


	print ldr_pmf.ProbGreater(0.5)

	#Updates the data-driven model with the test set
	Testdf = makedata('test_ldr.csv')
	ldr_pmf.Update(Testdf)
	
	#Getting the mean value of betas
	i=0
	# b0, b1, b2, sig
	params = [0,0,0,0]
	mean_params = [0,0,0,0]
	for values in ldr_pmf:
		params[0] += values[0]
		params[1] += values[1]
		params[2] += values[2]
		params[3] += values[3]
		i+=1

	mean_params[0] = params[0]/i
	mean_params[1] = params[1]/i
	mean_params[2] = params[2]/i
	mean_params[3] = params[3]/i

	print modelaccuracy_with_fixedbeta(Testdf, mean_params)

	#Comparing with the test 

if __name__ == '__main__':
    import sys
    main(*sys.argv)
