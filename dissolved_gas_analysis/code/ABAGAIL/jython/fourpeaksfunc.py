import sys
import os
import time

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer

from array import array



"""
Commandline parameter(s):
   none
"""


def fourpeaksfunc(N, iterations):

	rhcMult = 200
	saMult = 200
	gaMult = 2
	mimicMult = 1

	optimalOut = []
	timeOut = []
	evalsOut = []

	T=N/5
	fill = [2] * N
	ranges = array('i', fill)

	ef = FourPeaksEvaluationFunction(T)
	odd = DiscreteUniformDistribution(ranges)
	nf = DiscreteChangeOneNeighbor(ranges)
	mf = DiscreteChangeOneMutation(ranges)
	cf = SingleCrossOver()
	df = DiscreteDependencyTree(.1, ranges)
	hcp = GenericHillClimbingProblem(ef, odd, nf)
	gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
	pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

	for niter in iterations:

		iterOptimalOut = [N, niter]
		iterTimeOut = [N, niter]
		iterEvals = [N, niter]



		start = time.time()
		rhc = RandomizedHillClimbing(hcp)
		fit = FixedIterationTrainer(rhc, niter*rhcMult)
		fit.train()
		end = time.time()
		rhcOptimal = ef.value(rhc.getOptimal())
		rhcTime = end-start
		print "RHC optimum: " + str(rhcOptimal)
		print "RHC time: " + str(rhcTime)
		iterOptimalOut.append(rhcOptimal)
		iterTimeOut.append(rhcTime)
		functionEvals = ef.getNumEvals()
		ef.zeroEvals()
		iterEvals.append(functionEvals)
		 
		start = time.time()
		sa = SimulatedAnnealing(1E20, .8, hcp)
		fit = FixedIterationTrainer(sa, niter*saMult)
		fit.train()
		end = time.time()
		saOptimal = ef.value(sa.getOptimal())
		saTime = end-start
		print "SA optimum: " + str(saOptimal)
		print "SA time: " + str(saTime)
		iterOptimalOut.append(saOptimal)
		iterTimeOut.append(saTime)
		functionEvals = ef.getNumEvals()
		ef.zeroEvals()
		iterEvals.append(functionEvals)

		start = time.time()
		ga = StandardGeneticAlgorithm(200, 100, 10, gap)
		fit = FixedIterationTrainer(ga, niter*gaMult)
		fit.train()
		end = time.time()
		gaOptimal = ef.value(ga.getOptimal())
		gaTime = end - start
		print "GA optimum: " + str(gaOptimal)
		print "GA time: " + str(gaTime)
		iterOptimalOut.append(gaOptimal)
		iterTimeOut.append(gaTime)
		functionEvals = ef.getNumEvals()
		ef.zeroEvals()
		iterEvals.append(functionEvals)

		start = time.time()
		mimic = MIMIC(200, 20, pop)
		fit = FixedIterationTrainer(mimic, niter*mimicMult)
		fit.train()
		end = time.time()
		mimicOptimal = ef.value(mimic.getOptimal())
		mimicTime = end - start
		print "MIMIC optimum: " + str(mimicOptimal)
		print "MIMIC time: " + str(mimicTime)
		iterOptimalOut.append(mimicOptimal)
		iterTimeOut.append(mimicTime)
		functionEvals = ef.getNumEvals()
		ef.zeroEvals()
		iterEvals.append(functionEvals)
		
		optimalOut.append(iterOptimalOut)
		timeOut.append(iterTimeOut)
		evalsOut.append(iterEvals)
	
	return [optimalOut, timeOut, evalsOut]

