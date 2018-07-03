# traveling salesman algorithm implementation in jython
# This also prints the index of the points of the shortest route.
# To make a plot of the route, write the points at these indexes 
# to a file and plot them in your favorite tool.
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
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
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
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays

from array import array




"""
Commandline parameter(s):
    none
"""
def travelingsalesmanfunc(N, iterations):

	rhcMult = 1500
	saMult = 1500
	gaMult = 1
	mimicMult = 3


	random = Random()

	points = [[0 for x in xrange(2)] for x in xrange(N)]
	for i in range(0, len(points)):
		points[i][0] = random.nextDouble()
		points[i][1] = random.nextDouble()



	optimalOut = []
	timeOut = []
	evalsOut = []

	for niter in iterations:

		ef = TravelingSalesmanRouteEvaluationFunction(points)
		odd = DiscretePermutationDistribution(N)
		nf = SwapNeighbor()
		mf = SwapMutation()
		cf = TravelingSalesmanCrossOver(ef)
		hcp = GenericHillClimbingProblem(ef, odd, nf)
		gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)


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
		print "RHC Inverse of Distance: optimum: " + str(rhcOptimal)
		print "RHC time: " + str(rhcTime)
		#print "RHC Inverse of Distance: " + str(ef.value(rhc.getOptimal()))
		print "Route:"
		path = []
		for x in range(0,N):
			path.append(rhc.getOptimal().getDiscrete(x))
		print path
		iterOptimalOut.append(rhcOptimal)
		iterTimeOut.append(rhcTime)
		functionEvals = ef.getNumEvals()
		ef.zeroEvals()
		iterEvals.append(functionEvals)


		start = time.time()
		sa = SimulatedAnnealing(1E12, .999, hcp)
		fit = FixedIterationTrainer(sa, niter*saMult)
		fit.train()
		end = time.time()
		saOptimal = ef.value(sa.getOptimal())
		saTime = end-start
		print "SA Inverse of Distance optimum: " + str(saOptimal)
		print "SA time: " + str(saTime)
		#print "SA Inverse of Distance: " + str(ef.value(sa.getOptimal()))
		print "Route:"
		path = []
		for x in range(0,N):
			path.append(sa.getOptimal().getDiscrete(x))
		print path
		iterOptimalOut.append(saOptimal)
		iterTimeOut.append(saTime)
		functionEvals = ef.getNumEvals()
		ef.zeroEvals()
		iterEvals.append(functionEvals)



		start = time.time()
		ga = StandardGeneticAlgorithm(2000, 1500, 250, gap)
		fit = FixedIterationTrainer(ga, niter*gaMult)
		fit.train()
		end = time.time()
		gaOptimal = ef.value(ga.getOptimal())
		gaTime = end - start
		print "GA Inverse of Distance optimum: " + str(gaOptimal)
		print "GA time: " + str(gaTime)
		#print "GA Inverse of Distance: " + str(ef.value(ga.getOptimal()))
		print "Route:"
		path = []
		for x in range(0,N):
			path.append(ga.getOptimal().getDiscrete(x))
		print path
		iterOptimalOut.append(gaOptimal)
		iterTimeOut.append(gaTime)
		functionEvals = ef.getNumEvals()
		ef.zeroEvals()
		iterEvals.append(functionEvals)


		start = time.time()
		# for mimic we use a sort encoding
		ef = TravelingSalesmanSortEvaluationFunction(points);
		fill = [N] * N
		ranges = array('i', fill)
		odd = DiscreteUniformDistribution(ranges);
		df = DiscreteDependencyTree(.1, ranges); 
		pop = GenericProbabilisticOptimizationProblem(ef, odd, df);

		start = time.time()
		mimic = MIMIC(500, 100, pop)
		fit = FixedIterationTrainer(mimic, niter*mimicMult)
		fit.train()
		end = time.time()
		mimicOptimal = ef.value(mimic.getOptimal())
		mimicTime = end - start
		print "MIMIC Inverse of Distance optimum: " + str(mimicOptimal)
		print "MIMIC time: " + str(mimicTime)
		#print "MIMIC Inverse of Distance: " + str(ef.value(mimic.getOptimal()))
		print "Route:"
		path = []
		optimal = mimic.getOptimal()
		fill = [0] * optimal.size()
		ddata = array('d', fill)
		for i in range(0,len(ddata)):
			ddata[i] = optimal.getContinuous(i)
		order = ABAGAILArrays.indices(optimal.size())
		ABAGAILArrays.quicksort(ddata, order)
		print order
		iterOptimalOut.append(mimicOptimal)
		iterTimeOut.append(mimicTime)
		functionEvals = ef.getNumEvals()
		ef.zeroEvals()
		iterEvals.append(functionEvals)
	
		optimalOut.append(iterOptimalOut)
		timeOut.append(iterTimeOut)
		evalsOut.append(iterEvals)		
	
	return [optimalOut, timeOut, evalsOut]
