#!/bin/bash
# edit the classpath to to the location of your ABAGAIL jar file
#
export CLASSPATH=~/schoolJunk/ABAGAIL/ABAGAIL.jar:$CLASSPATH
mkdir -p data/plot logs image

# four peaks
echo "four peaks"
jython fourpeakswrapper.py

# knapsack
#echo "Running knapsack"
#jython knapsackwrapper.py

# traveling salesman
#echo "Running traveling salesman"
#jython travelingsalesmanwrapper.py
