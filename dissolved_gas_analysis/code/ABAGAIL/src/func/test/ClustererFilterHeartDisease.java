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
public class ClustererFilterHeartDisease {
	
	
	private static Instance[] instances = initializeInstances();
	
	private static int[] kVals = {2};//, 8, 16, 32};
	
	//private static String[] filterLabels = {"PCA e=2", "PCA e=3", "PCA e=4", "PCA e=5", "PCA e=6", "PCA e=7", "PCA e=8", "PCA e=9", "PCA e=10", "IndCA", "RPF", "InsCA", "No Filter"};
	private static String[] filterLabels = {"PCA e=2", "PCA e=3", "PCA e=4", "PCA e=5", "PCA e=6", "PCA e=7", "PCA e=8", "PCA e=9", "PCA e=10", 
											"IndCA e=2", "IndCA e=3", "IndCA e=4", "IndCA e=5", "IndCA e=6", "IndCA e=7", "IndCA e=8", "IndCA e=9", "IndCA e=10", 
											"RPF e=2", "RPF e=3", "RPF e=4", "RPF e=5", "RPF e=6", "RPF e=7", "RPF e=8", "RPF e=9", "RPF e=10", 
											"InsCA e=2", "InsCA e=3", "InsCA e=4", "InsCA e=5", "InsCA e=6", "InsCA e=7", "InsCA e=8", "InsCA e=8", "InsCA e=10", "No Filter"};
						
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
		DataSet set = new DataSet(instances);
        
        RandomOrderFilter randomOrder = new RandomOrderFilter();
        
        randomOrder.filter(set);	

         
        try {
            //Whatever the file path is.
            File name = new File("heartDiseaseClusterFilter.csv");
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
						sets[k] = set.copy();
					
					
					

					for (int k = 0; k < 9; k++)			
						filters[k] = new PrincipalComponentAnalysis(sets[k],k+2);
					for (int k = 0; k < 9; k++)
						filters[9+k] = new IndependentComponentAnalysis(sets[9+k],k+2);
					for (int k = 0; k < 9; k++)
						filters[18+k] = new RandomizedProjectionFilter(2+k,13);
					for (int k = 0; k < 9; k++)
						filters[27+k] = new InsignificantComponentAnalysis(sets[27+k],13);//LinearDiscriminantAnalysis(sets[3]);
					
					
        
			
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

        double[][][] attributes = new double[920][][];
        

        try {
            BufferedReader brData = new BufferedReader(new FileReader(new File("src/opt/test/heartDisease.data")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scanData = new Scanner(brData.readLine());
                scanData.useDelimiter(" ");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[13]; // 7 attributes
                //attributes[i][1] = new double[1];

                for(int j = 0; j < 13; j++)
                    attributes[i][0][j] = Double.parseDouble(scanData.next());

                //attributes[i][1][0] = Double.parseDouble(scan.next());
            }
            
            
				
            
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        
        
        try {
            BufferedReader brTarget = new BufferedReader(new FileReader(new File("src/opt/test/heartDisease.target")));

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
          
        double[] max = new double[13];
		
		for (int i = 0; i < 13; i++)
			max[i] = -1e20;
		
		for(int i = 0; i < attributes.length; i++) {
			for (int j = 0; j < 13; j++){
				if (attributes[i][0][j] >= max[j])
					max[j] = attributes[i][0][j];
			}
		}
		
		for(int i = 0; i < attributes.length; i++) {
			for (int j = 0; j < 13; j++){
				attributes[i][0][j] /= max[j];
			}
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
