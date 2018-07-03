package func.test;

import dist.Distribution;
import dist.MultivariateGaussian;
import func.KMeansClusterer;
import func.EMClusterer;
import shared.DataSet;
import shared.Instance;
import util.linalg.DenseVector;
import util.linalg.RectangularMatrix;

import util.linalg.Vector;


import shared.filt.*;//PrincipalComponentAnalysis;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Testing
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class ClustererFilterDGA {
	
	
	private static Instance[] instances = initializeInstances();
	
	private static int[] kVals = {2,3,4,5};//, 8, 16, 32};
	
	//private static String[] filterLabels = {"PCA e=1", "PCA e=2", "PCA e=3", "PCA e=4", "PCA e=5", "PCA e=6", "PCA e=7", "IndCA", "RPF", "InsCA", "No Filter"};
	private static String[] filterLabels = {"PCA e=2", "PCA e=3", "PCA e=4", "PCA e=5", "PCA e=6", "PCA e=7", 
											"IndCA e=2", "IndCA e=3", "IndCA e=4", "IndCA e=5", "IndCA e=6", "IndCA e=7",
											"RPF e=2", "RPF e=3", "RPF e=4", "RPF e=5", "RPF e=6", "RPF e=7", 
											"InsCA e=2", "InsCA e=3", "InsCA e=4", "InsCA e=5", "InsCA e=6", "InsCA e=7", "No Filter"};
						
	private static String[] clusterLabels = {"K-Means", "EM N=100", "EM N=1000", "EM N=10000"};
         	
	    /** 
     * The tolerance
     */
    private static final double TOLERANCE = 1E-6;
    /** 
     * The tolerance
     */
    private static final int MAX_ITERATIONS = 100;
	
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) throws Exception {
        
        DataSet set = new DataSet(instances);
        
        RandomOrderFilter randomOrder = new RandomOrderFilter();
        
        randomOrder.filter(set);
        
        System.out.println("Randomized dataset");
        
		//String[] filterLabels = {"Principal Component Analysis", "Independent Component Analysis", "Random Projection Filter", "Insignificant Component Analysis", "No Filter"};
			

         
        try {
            //Whatever the file path is.
            File name = new File("dgaCluster.csv");
            FileOutputStream is = new FileOutputStream(name);
            OutputStreamWriter osw = new OutputStreamWriter(is);    
            BufferedWriter w = new BufferedWriter(osw);
            
            System.out.println("Loaded file for writing");
            
            w.write("Clusterer,Filter,error,k,entropy");
            for (int i=0; i < kVals[kVals.length-1]; i++)
				w.write(",cluster"+String.valueOf(i+1));
            //for (int i=0; i < oaNames.length; i++)
			//	header += oaNames[i]+" Training Time,"+oaNames[i]+" Training Accuracy,"+oaNames[i]+" Testing Accuracy,";
            
            //header+="BackProp Training Time,BackProp Training Accuracy,BackProp Testing Accuracy";
            
            //w.write(header);
            w.newLine();


			for(int i = 0; i < kVals.length; i++) {
				
				EMClusterer em1 = new EMClusterer(kVals[i], TOLERANCE, 100); //c=1,2,3
				EMClusterer em2 = new EMClusterer(kVals[i], TOLERANCE, 100); //c=1,2,3
				EMClusterer em3 = new EMClusterer(kVals[i], TOLERANCE, 100); //c=1,2,3
				
				KMeansClusterer km = new KMeansClusterer(kVals[i]); //c=0
				
				System.out.println("Created clusterers for k="+String.valueOf(kVals[i]));
				
				for (int c = 0; c < clusterLabels.length; c++){
					
					ReversibleFilter[] filters = new ReversibleFilter[filterLabels.length-1];
        
					//ReversibleFilter[] emfilters = new ReversibleFilter[4];

					DataSet[] sets = new DataSet[filterLabels.length];//(instances);
					//DataSet[] emsets = new DataSet[5];
					
					
					
					for (int k = 0; k < filterLabels.length; k++)
						sets[k] = set.copy();
					
					System.out.println("Datsets copied");
					
					/*filters[0] = new PrincipalComponentAnalysis(sets[0]);					
					filters[1] = new IndependentComponentAnalysis(sets[1]);//,5);
					filters[2] = new RandomizedProjectionFilter(7,6);
					//kmfilters[2].filter(kmsets[2]);
					filters[3] = new InsignificantComponentAnalysis(sets[3],7);//LinearDiscriminantAnalysis(sets[3]);
					*/
					
					for (int k = 0; k < 6; k++)			
						filters[k] = new PrincipalComponentAnalysis(sets[k],k+2);
					System.out.println("Initialized PCA");
					for (int k = 0; k < 6; k++)
						filters[6+k] = new IndependentComponentAnalysis(sets[6+k],k+2);
					System.out.println("Initialized Independent CA");
					for (int k = 0; k < 6; k++)
						filters[12+k] = new RandomizedProjectionFilter(2+k,7);
					System.out.println("Initialized RPF");
					for (int k = 0; k < 6; k++){
						//System.out.println(k);
						filters[18+k] = new InsignificantComponentAnalysis(sets[18+k],k+2);//LinearDiscriminantAnalysis(sets[3]);
					}
					System.out.println("Initialized Insignificatn CA");
					
					
        
			
					System.out.println(clusterLabels[c]+ " k="+String.valueOf(kVals[i]));
					
					//w.write(clusterLabels[c]+ " k="+String.valueOf(kVals[i]));
					//w.newLine();

				
				
					for (int j = 0; j < filterLabels.length; j++){
						
						if (j < filterLabels.length-1)
							filters[j].filter(sets[j]);
		
						System.out.println("Completed filter: "+filterLabels[j]);
						
						if (c == 0)
							km.estimate(sets[j]);
						if (c == 1)
							em1.estimate(sets[j]);
						if (c == 2)
							em2.estimate(sets[j]);
						if (c == 3)
							em3.estimate(sets[j]);
						
						System.out.println("Completed clustering");				
			
						int[] clusterDist = new int[kVals[i]];
						for (int m = 0; m < kVals[i]; m++)
							clusterDist[m] = 0;
						//System.out(a)	
						Instance[] updatedInstances = new Instance[sets[j].size()];
						
						for (int k = 0; k < sets[j].size(); k++){
							Instance temp = sets[j].get(k);
							
							Vector tempData = temp.getData();
							double[] newData = new double[tempData.size()+1];
							for (int l=0; l < tempData.size(); l++)
								newData[l] = tempData.get(l);
							if (c == 0)
								newData[tempData.size()]=km.getClusterNumber(temp);
							if (c == 1)	
								newData[tempData.size()]=em1.getClusterNumber(temp);
							if (c == 2)	
								newData[tempData.size()]=em2.getClusterNumber(temp);
							if (c == 3)	
								newData[tempData.size()]=em3.getClusterNumber(temp);
							//tempData.add(tempData.size(),
							//Double clusterNum = km.getClusterNumber(temp); 
							//tempData.add(clusterNum);
							updatedInstances[k] = new Instance(new DenseVector(newData), temp.getLabel());
							
							clusterDist[(int)newData[tempData.size()]]++;
						}
						DataSet newDataSet = new DataSet(updatedInstances);
						
						
						//w.write(clusterLabels[c] + " k=" + String.valueOf(kVals[i])+". Filter ="+filterLabels[j]);
						//w.newLine();
						
						System.out.println(clusterLabels[c] + " k=" + String.valueOf(kVals[i])+". Filter ="+filterLabels[j]);
					
						//w.write(newDataSet.toString());
						//w.newLine();
						
						//System.out.println(newDataSet.toString());
						
						for (int m = 0; m < kVals[i]; m++){
							//w.write("Cluster "+String.valueOf(m)+" element size:"+String.valueOf(clusterDist[m]));
							//w.newLine();
							System.out.println("Cluster "+String.valueOf(m)+" element size:"+String.valueOf(clusterDist[m]));
						}
						
						double error = 0.0;
						if (c == 0)
							error = km.getError(sets[j]);
						if (c == 1)
							error = em1.getError(sets[j]);
						if (c == 2)
							error = em2.getError(sets[j]);
						if (c == 3)
							error = em3.getError(sets[j]);
						
						System.out.println("Clustering error ="+String.valueOf(error));
						
						double entropy = 0.0;
						for (int m = 0; m < kVals[i]; m++){
							double ds = ((double)clusterDist[m])/389.0*Math.log(((double)clusterDist[m])/389.0)/Math.log(2.0);
							entropy -= ds;
							//System.out.println(String.valueOf(clusterDist[m])+String.valueOf(ds));
						}
							
						w.write(clusterLabels[c]+","+filterLabels[j]+","+String.valueOf(error)+","+String.valueOf(kVals[i])+","+String.valueOf(entropy));
						for (int m=0; m < kVals[i]; m++){
							w.write(","+String.valueOf(clusterDist[m]));
						}
						w.newLine();
					}
					
				}
				
				
			}
            
            w.close();
        } catch (IOException e) {
            System.err.println("Problem writing to the file dgaCluster.csv");
        }
    }
    
    
    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[389][][];
        

        try {
            BufferedReader brData = new BufferedReader(new FileReader(new File("src/opt/test/dga.data")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scanData = new Scanner(brData.readLine());
                scanData.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[7]; // 7 attributes
                //attributes[i][1] = new double[1];

                for(int j = 0; j < 7; j++)
                    attributes[i][0][j] = Double.parseDouble(scanData.next());

                //attributes[i][1][0] = Double.parseDouble(scan.next());
            }
            
            
				
            
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        
        
        try {
            BufferedReader brTarget = new BufferedReader(new FileReader(new File("src/opt/test/dga.target")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scanTarget = new Scanner(brTarget.readLine());
                scanTarget.useDelimiter(",");

                attributes[i][1] = new double[1];

                attributes[i][1][0] = Double.parseDouble(scanTarget.next());
            }	
            
        }
        catch(Exception e) {
            e.printStackTrace();
        }
            
		double[] max = new double[7];
		
		for (int i = 0; i < 7; i++)
			max[i] = -1e20;
		
		for(int i = 0; i < attributes.length; i++) {
			for (int j = 0; j < 7; j++){
				if (attributes[i][0][j] >= max[j])
					max[j] = attributes[i][0][j];
			}
		}
		
		for(int i = 0; i < attributes.length; i++) {
			for (int j = 0; j < 7; j++){
				attributes[i][0][j] /= max[j];
			}
		}

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications are 0 or 1
            instances[i].setLabel(new Instance(attributes[i][1][0] == 1 ? 0 : 1));
        }

		System.out.println("Done loading data");
        return instances;
    }
    
    /*public static void normalizeAttributes(double[][][] attributes){
		double[] max = New double[7];
		
		for (int i = 0; i < 7; i++)
			max(i) = -1e20;
		
		for(int i = 0; i < attributes.length; i++) {
			for (j = 0; j < 7; j++){
				if (attributes[i][0][j] >= max(j))
					max(j) = attributes[i][0][j];
			}
		}
		
		for(int i = 0; i < attributes.length; i++) {
			for (j = 0; j < 7; j++){
				attributes[i][0][j] /= max(j))
			}
		}
	}*/		
}
