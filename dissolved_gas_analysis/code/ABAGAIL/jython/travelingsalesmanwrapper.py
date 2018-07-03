from travelingsalesmanfunc import travelingsalesmanfunc

#import matplotlib.pyplot as plt

N = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]#, 150, 200]

iterations = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

optimaOpen = open('travelingsalesmanOptima.csv','w')
timesOpen = open('travelingsalesmanTimes.csv','w')
evalsOpen = open('travelingsalesmanEvals.csv','w')


for n in N:
	optimalRHC = []
	timeRHC = []
	optimalSA = []
	timeSA = []
	optimalGA = []
	timeGA = []
	optimalMIMIC = []
	timeMIMIC = []
	print "!!!!!!!!!!!!!!!!!!!!!!Number of cities = " + str(n)
	[optima, times, evals] = travelingsalesmanfunc(n, iterations)

	for o in optima:
		optimaOpen.write(",".join(str(item) for item in o)+"\n")

	for t in times:
		timesOpen.write(",".join(str(item) for item in t)+"\n")
		
	for e in evals:	
		evalsOpen.write(",".join(str(item) for item in e)+"\n")

optimaOpen.close()
timesOpen.close()
evalsOpen.close()

#fig_tsp = plt.figure()
#ax = fig_tsp.add_subplot(1, 1, 1)
#ax.set_title("Traveling Salesman Problem Optima")
#ax.set_xlabel('Number of vertices')
#ax.set_ylabel('Inverse of distance for optimal path')
#ax.semilogy(iterations, trn_error, 'b', iterations, tst_error, 'g')
#ax.plot(N, optimalRHC, 'b', N, optimalSA, 'g', N, optimalGA, 'r', N, optimalMIMIC, 'g')
#ax.legend(['RHC', 'SA', 'GA', 'MIMIC'])#,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#ax.set_ylim([0,1])
#plt.show()
