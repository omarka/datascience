package opt.example;

import shared.Instance;

/**
 * An implementation of the traveling salesman problem
 * where the encoding used is a permutation of [0, ..., n]
 * where there are n+1 cities.  That is the encoding
 * is just the path to take.
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanRouteEvaluationFunction extends TravelingSalesmanEvaluationFunction {


    private int numCalls;

    /**
     * Make a new route evaluation function
     * @param points the points of the cities
     */
    public TravelingSalesmanRouteEvaluationFunction(double[][] points) {
        super(points);
        this.numCalls = 0;
    }

    /**
     * @see opt.EvaluationFunction#value(opt.OptimizationData)
     */
    public double value(Instance d) {
		this.numCalls++;
        double distance = 0;
        for (int i = 0; i < d.size() - 1; i++) {
            distance += getDistance(d.getDiscrete(i), d.getDiscrete(i+1));
        }
        distance += getDistance(d.getDiscrete(d.size() - 1), d.getDiscrete(0));
        return 1/distance;
    }

    public int getNumEvals(){
		return this.numCalls;
	}

	
	public void zeroEvals(){
		this.numCalls = 0;
	}


}
