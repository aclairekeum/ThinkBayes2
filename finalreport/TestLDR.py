import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import thinkbayes2
import thinkplot

LDRcsv = 'ldr.csv'
#file name for data

Prior_Probability = 0.5

class LDR(thinkbayes2.Suite, thinkbayes2.Joint):
	def Likelihood(self, data, hypo):

		beta0, beta1, beta2, sigma = hypo
		total_likelihood = 1
		for i, row in enumerate(data.values):
			meetups = row[2]
			talks = row[1]
			breakup = row[0]
			log_breakup_odds = beta0 + beta1 * meetups + beta2 * talks
			breakup_prob = logo_to_p(log_breakup_odds, 0.5)
			error = breakup - breakup_prob
			total_likelihood *= thinkbayes2.EvalNormalPdf(error, mu=0, sigma=sigma)
			return total_likelihood

def makedata(filename):
	"Returns dataframe from csv file"
	df = pd.read_csv(filename,
					names=['relationship','meet','talk'],
					header=0)
	return df

def logo_to_p(logo, prior_p):
	o = np.exp(logo)
	prior_odds = prior_p/(1-prior_p)
	post_odds = prior_odds * o
	p = post_odds/(post_odds+1)

	return p

def pmf_from_data(filename, params):

	LDRdf = makedata(filename)

	b0 = params[0]
	b1 = params[1]
	b2 = params[2]

	est_range = 0.05
	b0hypos = np.linspace(b0*(1-est_range), b0*(1+est_range),10)
	b1hypos = np.linspace(b1*(1-est_range), b1*(1+est_range),10)
	b2hypos = np.linspace(b2*(1-est_range), b2*(1+est_range),10)
	sigmahypos = np.linspace(0.001,0.05,10)

	hypos = [(b0hypo,b1hypo,b2hypo,sigmahypo) for b0hypo in b0hypos for b1hypo in b1hypos for b2hypo in b2hypos for sigmahypo in sigmahypos]
	ldr_pmf = LDR(hypos)
	ldr_pmf.Update(LDRdf)

	return ldr_pmf

def main(script):

	beta = [0, 0, 0]

	df = makedata(LDRcsv)
	formula = 'relationship ~ meet+talk'
	results = smf.logit(formula, data=df).fit()
	print results.summary()

	beta[0] = results.params['Intercept']
	beta[1] = results.params['meet']
	beta[2] = results.params['talk']

	ldr_pmf = pmf_from_data(LDRcsv,beta)

	"The code below was taken from cypressf/thinkbayes2 to calculate and plot the maximum Likelihood"

	maximum_likelihoods = [0, 0, 0, 0]
	for title, i in [('b0', 0), ('b1', 1), ('b2', 2),('sigma',3)]:
		marginal = ldr_pmf.Marginal(i)
		maximum_likelihoods[i] = marginal.MaximumLikelihood()
		thinkplot.Hist(marginal)
		plt.title("PMF for " + title)
		plt.show()

if __name__ == '__main__':
    import sys
    main(*sys.argv)
