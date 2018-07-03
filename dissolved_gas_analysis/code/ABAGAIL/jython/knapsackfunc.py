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
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
from array import array




"""
Commandline parameter(s):
    none
"""
def knapsackfunc(NUM_ITEMS,  iterations):


        rhcMult = 600
        saMult = 600
        gaMult = 4
        mimicMult = 3


	# Random number generator */
	random = Random()
	# The number of items
	#NUM_ITEMS = 40
	# The number of copies each
	COPIES_EACH = 4
	# The maximum weight for a single element
	MAX_WEIGHT = 50
	# The maximum volume for a single element
	MAX_VOLUME = 50
	# The volume of the knapsack 
	KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

	# create copies
	fill = [COPIES_EACH] * NUM_ITEMS
	copies = array('i', fill)

	# create weights and volumes
	fill = [0] * NUM_ITEMS
	weights = array('d', fill)
	volumes = array('d', fill)
	for i in range(0, NUM_ITEMS):
		weights[i] = random.nextDouble() * MAX_WEIGHT
		volumes[i] = random.nextDouble() * MAX_VOLUME


	# create range
	fill = [COPIES_EACH + 1] * NUM_ITEMS
	ranges = array('i', fill)

	ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
	odd = DiscreteUniformDistribution(ranges)
	nf = DiscreteChangeOneNeighbor(ranges)
	mf = DiscreteChangeOneMutation(ranges)
	cf = UniformCrossOver()
	df = DiscreteDependencyTree(.1, ranges)
	hcp = GenericHillClimbingProblem(ef, odd, nf)
	gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
	pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

	optimalOut = []
	timeOut = []
	evalsOut = []

	for niter in iterations:

		iterOptimalOut = [NUM_ITEMS, niter]
		iterTimeOut = [NUM_ITEMS, niter]
		iterEvals = [NUM_ITEMS, niter]



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
		sa = SimulatedAnnealing(100, .95, hcp)
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
		ga = StandardGeneticAlgorithm(200, 150, 25, gap)
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
		mimic = MIMIC(200, 100, pop)
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
