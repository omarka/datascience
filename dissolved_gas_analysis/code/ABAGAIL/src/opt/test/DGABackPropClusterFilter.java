package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.Arrays;
import java.util.Scanner;
import java.util.Random;
import java.io.*;
import java.text.*;

import shared.ConvergenceTrainer;
import shared.DataSet;
import shared.Instance;
import shared.SumOfSquaresError;

import func.KMeansClusterer;
import func.EMClusterer;

import shared.filt.*;

import util.linalg.Vector;
import util.linalg.DenseVector;

/**
 * Implementation of randomized hill climbing, simulated annealing, genetic algorithm, and back-propagation to
 * find optimal weights to a neural network that is classifying dissolved gas analysis measurements as having either 
 * indicating an arcing transformer, or not indicating an arcing transformer. 
 * 
 * This is a modification of the pre-exisiting Abalone classification problem
 *
 * @author Omar Ahmed
 * @version 1.0
 */
public class DGABackPropClusterFilter {
    private static Instance[] instances = initializeInstances();
    
    private static Instance[][] clusteredInstances;
    
    private static Instance[][] trainingInstances = new Instance[3][311];
    
    private static Instance[][] testingInstances = new Instance[3][78];

    private static int inputLayer = 7, hiddenLayer = 10, outputLayer = 1;
    
    private static int trainingSize = (int)(389*0.8), testingSize = 389-(int)(389*0.8);
    
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

	private static DataSet set = new DataSet(instances);

    private static DataSet[] clusteredSets = new DataSet[3];

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    //private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[14];

    private static String[] clusterNames = {"k-Means", "EM", "None"};
    private static String[] filterNames = {"Principal Component Analysis", "Independent Component Analysis", "Random Projection Filter", "Insignificant Component Analysis","None"};
		
	private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
		
		int[] iterations = {10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000};
		
		int kVal = 2;
		
		double[] data;
		
		String header = "Iterations,";

		System.out.println("Staring ANN");

        
        RandomOrderFilter randomOrder = new RandomOrderFilter();
        
        randomOrder.filter(set);
        

		clusterData();

		//splitTrainingTesting();
		
		try {
            //Whatever the file path is.
            File name = new File("dgaNNcluster.csv");
            FileOutputStream is = new FileOutputStream(name);
            OutputStreamWriter osw = new OutputStreamWriter(is);    
            BufferedWriter w = new BufferedWriter(osw);
            

            header="";
            for (int i=0; i < 3; i++)
				for (int j = 0; j < 5; j++)
					header += clusterNames[i]+" " + filterNames[j]+" " +" Training Time,"+clusterNames[i]+" " + filterNames[j]+" " +" Training Accuracy,"+clusterNames[i]+" " + filterNames[j]+" " +" Testing Accuracy,";
           // header+="UnClustered Training Time,UnClustered Training Accuracy,UnClustered Testing Accuracy";
            
            w.write(header);
            w.newLine();
            
         

            
         	for (int i=0; i < iterations.length; i++){
				data = getData(iterations[i]);
				String s = Arrays.toString(data);
				
				// cut off the square brackets at the beginning and at the end
				s = s.substring(1, s.length() - 1);
				w.write(String.valueOf(iterations[i]));
				w.write(",");
				w.write(s);
				
				w.newLine();
			}
		
            
            w.close();
        } catch (IOException e) {
            System.err.println("Problem writing to the file dgaOut.csv");
        }
	
	}
	
	private static double[] getData(int iterationNum){
		
		
		double start, end, trainingTime, testingTime, predicted, actual, trash, correct, incorrect;
		
		double[] data = new double[(clusterNames.length)*(filterNames.length)*3];
		
		
        for(int i = 0; i < clusteredSets.length; i++) {
			if (i < 2)
				inputLayer = 8;
			else
				inputLayer = 7;
			
			DataSet[] filteredSets = new DataSet[5];
			ReversibleFilter[] filters = new ReversibleFilter[4];
				
			for (int k = 0; k < 5; k++)
				filteredSets[k] = clusteredSets[i].copy();
        
			filters[0] = new PrincipalComponentAnalysis(filteredSets[0]);
			filters[1] = new IndependentComponentAnalysis(filteredSets[1]);//,5);
			filters[2] = new RandomizedProjectionFilter(7,7);
			filters[2].filter(filteredSets[2]);
			filters[3] = new InsignificantComponentAnalysis(filteredSets[3]);//LinearDiscriminantAnalysis(sets[3]);   
		
			//System.out.println("Done with filters");
				
			for(int j = 0; j < filteredSets.length; j++) {
			
				TestTrainSplitFilter ttsf = new TestTrainSplitFilter(80);
			
				ttsf.filter(filteredSets[i]);
			
				DataSet trainingSet = ttsf.getTrainingSet();
				DataSet testingSet = ttsf.getTestingSet();	
            
				//System.out.println("Done with training-testing split");
            
				BackPropagationNetwork network = new BackPropagationNetwork();
				network = factory.createClassificationNetwork(
					new int[] {inputLayer, hiddenLayer, outputLayer});
				 
				FixedIterationTrainer bpTrainer = new FixedIterationTrainer(
					   new BatchBackPropagationTrainer(trainingSet, network,
						   new SumOfSquaresError(), new RPROPUpdateRule()), iterationNum);
				
				start = System.nanoTime();
				
				bpTrainer.train();
				
				//System.out.println("cluster "+clusterNames[i]+" filter "+filterNames[j]+"trained neural net");
				
				end = System.nanoTime();
				trainingTime = (end - start)/Math.pow(10,9);
				
				data[i*3+j*5] = trainingTime;     
				
				//System.out.println(trainingSet);
				
				correct = 0;
				incorrect = 0;
				for(int k = 0; k < trainingSize; k++) {

					network.setInputValues(trainingSet.get(k).getData());
					network.run();
					
					predicted = Double.parseDouble(trainingSet.get(k).getLabel().toString());
					actual = Double.parseDouble(network.getOutputValues().toString());

					if (Math.abs(predicted - actual) < 0.5)
						correct++;
					else
						incorrect++;

				}
				
				data[i*3+j*5+1] = correct/(correct+incorrect)*100.0;         //Training Accuracy
				
				correct = 0;
				incorrect = 0;
				start = System.nanoTime();
				for(int k = 0; k < testingSize; k++) {
					network.setInputValues(testingSet.get(k).getData());
					network.run();

					predicted = Double.parseDouble(testingSet.get(k).getLabel().toString());
					actual = Double.parseDouble(network.getOutputValues().toString());

					if (Math.abs(predicted - actual) < 0.5)
						correct++;
					else
						incorrect++;

				}
				end = System.nanoTime();
				testingTime = (end - start)/Math.pow(10,9);
				
				
				//System.out.println("Back Propagation Convergence in " 
				//    + bpTrainer.getIterations() + " iterations");
					
				results +=  "\nResults for Back Propagation + cluster "+clusterNames[i]+" filter "+filterNames[j]+": \nCorrectly classified " + correct + 
								" instances. \nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
								+ df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
								+ " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";    
								
				System.out.println(results);
				
				data[i*3+j*5+2] = correct/(correct+incorrect)*100.0; //Testing Accuracy          
			}
            
            
        }
		return data;

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
        
        
        //DataSet[] sets = new DataSet[5];//(instances);
        
        

		//set = new DataSet(trainingInstances);
		
		//for (int i=1; i<100; i++)
		//	shuffleArray(instances);
		
		//clusterData();
		
        return instances;
    }
    
	
	private static void clusterData(){
		
		int kVal = 2;
		
		double TOLERANCE = 1E-6;
    /** 
     * The tolerance
     */
		int MAX_ITERATIONS = 1000;		
		
		KMeansClusterer km = new KMeansClusterer(kVal);
		EMClusterer em = new EMClusterer(kVal, TOLERANCE, MAX_ITERATIONS);
		
		for (int i = 0; i < 3; i++)		
			clusteredSets[i] = set.copy();
		
		km.estimate(clusteredSets[0]);
		em.estimate(clusteredSets[1]);
		
		
		Instance[][] clusteredInstances = new Instance[3][clusteredSets[0].size()];
		
		for (int i = 0; i < 2; i++){
						
			int[] clusterDist = new int[kVal];
			for (int m = 0; m < kVal; m++)
				clusterDist[m] = 0;
				
			//Instance[] updatedInstances = new Instance[clusteredSets[i].size()];
			
			
			for (int k = 0; k < clusteredSets[i].size(); k++){
				Instance temp = clusteredSets[i].get(k);
							
				Vector tempData = temp.getData();
				double[] newData = new double[tempData.size()+1];
				for (int l=0; l < tempData.size(); l++)
					newData[l] = tempData.get(l);
				if (k == 0)
					newData[tempData.size()]=km.getClusterNumber(temp);
				if (k == 1)
					newData[tempData.size()]=em.getClusterNumber(temp);		

				//updatedInstances[k] = new Instance(new DenseVector(newData), temp.getLabel());
				clusteredInstances[i][k] = new Instance(new DenseVector(newData), temp.getLabel());
							
				clusterDist[(int)newData[tempData.size()]]++;
			}
			clusteredSets[i] = new DataSet(clusteredInstances[i]);
			//clusteredInstances[i] = updatedInstances;
			System.out.println(clusteredInstances[0][0]);//clusteredSets[i]);
			
			
			for (int m = 0; m < kVal; m++){
				System.out.println(clusterNames[i] + " Cluster "+String.valueOf(m)+" element size:"+String.valueOf(clusterDist[m]));
				//w.newLine();
			}					
		}
		//return clusteredInstances;
	}	
   /* 
    private static void splitTrainingTesting()
    {		

		//int seed = System.nanoTime();

        //Random rand = new Random(0);   // create seeded number generator
        
        System.out.println(clusteredSets[0].getInstances()[0]);
        
		//Instance[][] trainingInstances = new Instance[3][trainingSize];
		//Instance[][] testingInstances = new Instance[3][testingSize];
        
        for (int i = 0; i < 3; i++){
			
			System.out.println("Splitting cluster "+String.valueOf(i));

			for (int j = 0; j < trainingSize; j++){

				//System.out.print(String.valueOf(i)+" "+String.valueOf(j));
				Instance temp = clusteredSets[i].getInstances()[j];
				trainingInstances[i][j] = new Instance((temp).getData(),temp.getLabel());//,clusteredInstances[i][j].getWeight);

				//trainingInstances[i][j] = new Instance((clusteredInstances[i][j]).getData(),clusteredInstances[i][j].getLabel());//,clusteredInstances[i][j].getWeight);
			}
				
			for (int j = 0; j < testingSize; j++){
				int k = j+trainingSize;
				
				Instance temp = clusteredSets[i].getInstances()[k];
				testingInstances[i][j] = new Instance(temp.getData(),temp.getLabel());//,clusteredInstances[i][k].getWeight);
			}
			//System.arraycopy(clusteredInstances[i], 0, trainingInstances[i], 0, trainingSize);
			System.out.println("Finished array copy 2");
			//System.arraycopy(clusteredInstances[i], trainingSize, testingInstances[i], 0, testingSize);
			//System.out.println("Finished array copy 3");
		}
		
		//return testingInstances;
	}
	
	// Shuffle helper fucntion
	private static void shuffleArray(Instance[] temp)
	{
		Random rnd = new Random();
		for (int i = temp.length - 1; i > 0; i--)
		{
			int index = rnd.nextInt(i + 1);
			// Simple swap
			Instance a = temp[index];
			temp[index] = temp[i];
			temp[i] = a;
		}
	}
    private static void randomizeData()
	{
		for (int i=1; i<100; i++)
			shuffleArray(instances);
			
	}
	*/

  
			
    
}
