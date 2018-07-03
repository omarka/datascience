package func.test;

import dist.Distribution;
import dist.MultivariateGaussian;
import func.EMClusterer;
import shared.DataSet;
import shared.Instance;
import util.linalg.DenseVector;
import util.linalg.RectangularMatrix;

/**
 * Testing
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class EMClustererTest {
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) throws Exception {
        Instance[] instances = new Instance[100];
        MultivariateGaussian mga = new MultivariateGaussian(new DenseVector(new double[] {100, 100, 100}), RectangularMatrix.eye(3).times(.01)); 
        MultivariateGaussian mgb = new MultivariateGaussian(new DenseVector(new double[] {-1, -1, -1}), RectangularMatrix.eye(3).times(10)); 
         for (int i = 0; i < instances.length; i++) {
            if (Distribution.random.nextBoolean()) {
                instances[i] = mga.sample(null);   
            } else {
                instances[i] = mgb.sample(null);
            }
        }
        DataSet set = new DataSet(instances);
        EMClusterer em = new EMClusterer();
        em.estimate(set);
        System.out.println(em);
                
        int[] clusterDist = new int[2];
		for (int m = 0; m < 2; m++)
			clusterDist[m] = 0;
			
        for (int k=0; k < instances.length; k++)
			clusterDist[em.getClusterNumber(instances[k])]++;
		
		for (int m = 0; m < 2; m++)
			System.out.println("Cluster "+String.valueOf(m)+" element size:"+String.valueOf(clusterDist[m]));
		
		
    }
}
