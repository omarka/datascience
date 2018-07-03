package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

import shared.ConvergenceTrainer;
import shared.DataSet;
import shared.Instance;
import shared.SumOfSquaresError;

import shared.filt.*;

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
public class DGABackPropFilter {
    private static Instance[] instances = initializeInstances();
    
    private static Instance[] trainingInstances, testingInstances;

    private static int inputLayer = 7, hiddenLayer = 10, outputLayer = 1;
    
    private static int trainingSize = 311, testingSize = 78;
    
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet[] filteredSets = new DataSet[5];

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[5];
    //private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[14];

    private static ReversibleFilter[] filters = new ReversibleFilter[4];
    private static String[] rfNames = {"Principal Component Analysis", "Independent Component Analysis", "Random Projection Filter", "Insignificant Component Analysis"};
		
	private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
		
		int[] iterations = {10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000};
		
		double[] data;
		
		String header = "Iterations,";

		System.out.println("Staring ANN");


		//splitTrainingTesting();
		
		for (int k = 0; k < 5; k++)
			filteredSets[k] = new DataSet(instances);
        
        filters[0] = new PrincipalComponentAnalysis(filteredSets[0]);
        filters[1] = new IndependentComponentAnalysis(filteredSets[1]);//,5);
        filters[2] = new RandomizedProjectionFilter(7,7);
        filters[2].filter(filteredSets[2]);
        filters[3] = new InsignificantComponentAnalysis(filteredSets[3]);//LinearDiscriminantAnalysis(sets[3]);   
		
		

		System.out.println("Done with filters");	

		 try {
            //Whatever the file path is.
            File name = new File("dgaNNfilter.csv");
            FileOutputStream is = new FileOutputStream(name);
            OutputStreamWriter osw = new OutputStreamWriter(is);    
            BufferedWriter w = new BufferedWriter(osw);
            

            header="";
            for (int i=0; i < rfNames.length; i++)
				header += rfNames[i]+" Training Time,"+rfNames[i]+" Training Accuracy,"+rfNames[i]+" Testing Accuracy,";
            header+="Unfiltered Training Time,Unfiltered Training Accuracy,Unfiltered Testing Accuracy";
            
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
		
		double[] data = new double[(filteredSets.length)*3];
		
		
		//ReversibleFilter[] filters = new ReversibleFilter[4];

		
        for(int i = 0; i < filteredSets.length; i++) {
			
			TestTrainSplitFilter ttsf = new TestTrainSplitFilter(80);
			
			ttsf.filter(filteredSets[i]);
			
			DataSet trainingSet = ttsf.getTrainingSet();
			DataSet testingSet = ttsf.getTestingSet();
			

			System.out.println("Done with data split");
			
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
			 
			 FixedIterationTrainer bpTrainer = new FixedIterationTrainer(
				   new BatchBackPropagationTrainer(trainingSet, networks[i],
					   new SumOfSquaresError(), new RPROPUpdateRule()), iterationNum);
			
			start = System.nanoTime();
			
			bpTrainer.train();

			System.out.println("Done with training");
			
			end = System.nanoTime();
			trainingTime = (end - start)/Math.pow(10,9);
			
			data[i*3] = trainingTime;     
			
			correct = 0;
			incorrect = 0;
			for(int j = 0; j < trainingSize; j++) {
				System.out.println(trainingSet.get(j).getData());
				networks[i].setInputValues(trainingSet.get(j).getData());
				networks[i].run();

				predicted = Double.parseDouble(trainingSet.get(j).getLabel().toString());
				actual = Double.parseDouble(networks[i].getOutputValues().toString());

				if (Math.abs(predicted - actual) < 0.5)
					correct++;
				else
					incorrect++;

			}
			
			data[i*3+1] = correct/(correct+incorrect)*100.0;         //Training Accuracy
			

			System.out.println("Calculated training accuracy");
			
			correct = 0;
			incorrect = 0;
			start = System.nanoTime();
			for(int j = 0; j < testingSize; j++) {
				networks[i].setInputValues(testingSet.get(j).getData());
				networks[i].run();

				predicted = Double.parseDouble(testingSet.get(j).getLabel().toString());
				actual = Double.parseDouble(networks[i].getOutputValues().toString());

				if (Math.abs(predicted - actual) < 0.5)
					correct++;
				else
					incorrect++;

			}
			end = System.nanoTime();
			testingTime = (end - start)/Math.pow(10,9);
			
			
			//System.out.println("Back Propagation Convergence in " 
			//    + bpTrainer.getIterations() + " iterations");
				
			results +=  "\nResults for Back Propagation + : \nCorrectly classified " + correct + 
							" instances. \nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
							+ df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
							+ " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";    
							
			System.out.println(results);
			
			data[i*3+2] = correct/(correct+incorrect)*100.0; //Testing Accuracy          
            
            
            
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

		//randData.randomize(new java.util.Random(0));         // randomize data with number generator
        for (int i=1; i<100; i++)
			shuffleArray(instances);

        return instances;
    }
    
    private static void splitTrainingTesting()
    {		

		//int seed = System.nanoTime();

        //Random rand = new Random(0);   // create seeded number generator
        
        
        
		Instance[] randData = new Instance[389];   // create copy of original data
        
        System.arraycopy(instances, 0, randData, 0, 389);
        
        System.out.println("Finished array copy 1");
        
		//randData.randomize(new java.util.Random(0));         // randomize data with number generator
        for (int i=1; i<10; i++)
			shuffleArray(randData);
        
        trainingInstances = new Instance[trainingSize];
		testingInstances = new Instance[testingSize];

		System.arraycopy(randData, 0, trainingInstances, 0, trainingSize);
        System.out.println("Finished array copy 2");
		System.arraycopy(randData, trainingSize, testingInstances, 0, testingSize);
        System.out.println("Finished array copy 3");
		
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
  
			
    
}
