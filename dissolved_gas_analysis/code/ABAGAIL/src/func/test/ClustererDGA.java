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
public class ClustererDGA {
	
	
	private static Instance[] instances = initializeInstances();
	
	private static int[] kVals = {2, 3};//, 8, 16, 32};
	
	//private static String[] filterLabels = {"PCA e=1", "PCA e=2", "PCA e=3", "PCA e=4", "PCA e=5", "PCA e=6", "PCA e=7", "IndCA", "RPF", "InsCA", "No Filter"};
	private static String[] filterLabels = {"PCA", "IndCA", "RPF", "InsCA", "No Filter"};
						
	private static String[] clusterLabels = {"K-Means Cluster", "EM Cluster"};
         	
	    /** 
     * The tolerance
     */
    private static final double TOLERANCE = 1E-9;
    /** 
     * The tolerance
     */
    private static final int MAX_ITERATIONS = 100000;
	
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) throws Exception {
        
        //DataSet set = new DataSet(instances);
        
        
		//String[] filterLabels = {"Principal Component Analysis", "Independent Component Analysis", "Random Projection Filter", "Insignificant Component Analysis", "No Filter"};
			

         
        try {
            //Whatever the file path is.
            File name = new File("dgaCluster.csv");
            FileOutputStream is = new FileOutputStream(name);
            OutputStreamWriter osw = new OutputStreamWriter(is);    
            BufferedWriter w = new BufferedWriter(osw);
            
            //for (int i=0; i < oaNames.length; i++)
			//	header += oaNames[i]+" Training Time,"+oaNames[i]+" Training Accuracy,"+oaNames[i]+" Testing Accuracy,";
            
            //header+="BackProp Training Time,BackProp Training Accuracy,BackProp Testing Accuracy";
            
            //w.write(header);
            //w.newLine();


			for(int i = 0; i < kVals.length; i++) {
				
				EMClusterer em = new EMClusterer(kVals[i], TOLERANCE, MAX_ITERATIONS); //c=1
				
				KMeansClusterer km = new KMeansClusterer(kVals[i]); //c=0
				
				for (int c = 0; c < 2; c++){
					
					ReversibleFilter[] filters = new ReversibleFilter[filterLabels.length-1];
        
					//ReversibleFilter[] emfilters = new ReversibleFilter[4];

					DataSet[] sets = new DataSet[filterLabels.length];//(instances);
					//DataSet[] emsets = new DataSet[5];
					
					
					
					for (int k = 0; k < filterLabels.length; k++)
						sets[k] = new DataSet(instances);
					
					
					
					filters[0] = new PrincipalComponentAnalysis(sets[0]);					
					filters[1] = new IndependentComponentAnalysis(sets[1]);//,5);
					filters[2] = new RandomizedProjectionFilter(7,6);
					//kmfilters[2].filter(kmsets[2]);
					filters[3] = new InsignificantComponentAnalysis(sets[3],7);//LinearDiscriminantAnalysis(sets[3]);
					
					
        
			
					System.out.println(clusterLabels[c]+ " k="+String.valueOf(kVals[i]));
					
					w.write(clusterLabels[c]+ " k="+String.valueOf(kVals[i]));
					w.newLine();

				
				
					for (int j = 0; j < filterLabels.length; j++){
						
						if (j < filterLabels.length-1)
							filters[j].filter(sets[j]);
		
						System.out.println("Completed filter: "+filterLabels[j]);
						
						if (c == 0)
							km.estimate(sets[j]);
						else
							em.estimate(sets[j]);
						
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
							else	
								newData[tempData.size()]=em.getClusterNumber(temp);
							//tempData.add(tempData.size(),
							//Double clusterNum = km.getClusterNumber(temp); 
							//tempData.add(clusterNum);
							updatedInstances[k] = new Instance(new DenseVector(newData), temp.getLabel());
							
							clusterDist[(int)newData[tempData.size()]]++;
						}
						DataSet newDataSet = new DataSet(updatedInstances);
						
						
						w.write(clusterLabels[c] + " k=" + String.valueOf(kVals[i])+". Filter ="+filterLabels[j]);
						w.newLine();
						
						System.out.println(clusterLabels[c] + " k=" + String.valueOf(kVals[i])+". Filter ="+filterLabels[j]);
					
						//w.write(newDataSet.toString());
						//w.newLine();
						
						//System.out.println(newDataSet.toString());
						
						for (int m = 0; m < kVals[i]; m++){
							w.write("Cluster "+String.valueOf(m)+" element size:"+String.valueOf(clusterDist[m]));
							w.newLine();
							System.out.println("Cluster "+String.valueOf(m)+" element size:"+String.valueOf(clusterDist[m]));
						}
						
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
            

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications are 0 or 1
            instances[i].setLabel(new Instance(attributes[i][1][0] == 1 ? 0 : 1));
        }

        return instances;
    }
}
