from fourpeaksfunc import fourpeaksfunc

N = [10, 20, 30, 40, 50]


#iterationMax = 10000.0
iterations = [1000, 5000, 10000, 50000, 100000, 500000, 100000]

optimaOpen = open('fourpeaksOptima.csv','w')
timesOpen = open('fourpeaksTimes.csv','w')
evalsOpen = open('fourpeaksEvals.csv','w')

for n in N:
	print "!!!!!!!!!!!!!!!!!!!!!!Input Size = " + str(n)
	[optima, times, evals] = fourpeaksfunc(n, iterations)
	for o in optima:
		optimaOpen.write(",".join(str(item) for item in o)+"\n")

	for t in times:
		timesOpen.write(",".join(str(item) for item in t)+"\n")
		
	for e in evals:	
		evalsOpen.write(",".join(str(item) for item in e)+"\n")

optimaOpen.close()
timesOpen.close()
evalsOpen.close()



