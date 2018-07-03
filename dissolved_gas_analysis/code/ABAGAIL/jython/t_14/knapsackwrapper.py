from knapsackfunc import knapsackfunc

N = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]#, 150, 200]

iterations = [1000, 2000,  3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

optimaOpen = open('knapsackOptima.csv','w')
timesOpen = open('knapsackTimes.csv','w')
evalsOpen = open('knapsackEvals.csv','w')

for n in N:
	print "!!!!!!!!!!!!!!!!!!!!!!Number of items = " + str(n)
#	print "Number of Iterations = " + str(int(i*iterationMax[0]))
	[optima, times, evals] = knapsackfunc(n, iterations)
	for o in optima:
		optimaOpen.write(",".join(str(item) for item in o)+"\n")

	for t in times:
		timesOpen.write(",".join(str(item) for item in t)+"\n")
		
	for e in evals:	
		evalsOpen.write(",".join(str(item) for item in e)+"\n")



optimaOpen.close()
timesOpen.close()
evalsOpen.close()



