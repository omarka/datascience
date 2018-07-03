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
public class DGATest {
    private static Instance[] instances = initializeInstances();
    
    private static Instance[] trainingInstances, testingInstances;

    private static int inputLayer = 7, hiddenLayer = 10, outputLayer = 1;
    
    private static int trainingSize = 311, testingSize = 78;
    
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set;

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[14];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[14];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[14];
    private static String[] oaNames = {"RHC", "SA T=1e12 r=0.95", "SA T=1e9 r=0.95",  "SA T=1e6 r=0.95",  "SA T=1e12 r=0.8", "SA T=1e9 r=0.8",  "SA T=1e6 r=0.8",  "GA p=200 s=200 m=10",  "GA p=200 s=100 m=10",  "GA p=100 s=100 m=10",  "GA p=100 s=50 m=10",  "GA p=50 s=50 m=10",  "GA p=100 s=100 m=5",  "GA p=100 s=100 m=20"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
		
		int[] iterations = {10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000};
		
		double[] data;
		
		String header = "Iterations,";

		 try {
            //Whatever the file path is.
            File name = new File("dgaOut.csv");
            FileOutputStream is = new FileOutputStream(name);
            OutputStreamWriter osw = new OutputStreamWriter(is);    
            BufferedWriter w = new BufferedWriter(osw);
            
            for (int i=0; i < oaNames.length; i++)
				header += oaNames[i]+" Training Time,"+oaNames[i]+" Training Accuracy,"+oaNames[i]+" Testing Accuracy,";
            
            header+="BackProp Training Time,BackProp Training Accuracy,BackProp Testing Accuracy";
            
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
		
		double[] data = new double[(oa.length+1)*3];
		
		splitTrainingTesting();
		
		set = new DataSet(trainingInstances);
		
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        
        oa[1] = new SimulatedAnnealing(1E12, .95, nnop[1]);
        oa[2] = new SimulatedAnnealing(1E9, .95, nnop[2]);
        oa[3] = new SimulatedAnnealing(1E6, .95, nnop[3]);
        oa[4] = new SimulatedAnnealing(1E12, .8, nnop[4]);
        oa[5] = new SimulatedAnnealing(1E9, .8, nnop[5]);
        oa[6] = new SimulatedAnnealing(1E6, .8, nnop[6]);
        
        oa[7] = new StandardGeneticAlgorithm(200, 200, 10, nnop[7]);
        oa[8] = new StandardGeneticAlgorithm(200, 100, 10, nnop[8]);
        oa[9] = new StandardGeneticAlgorithm(100, 100, 10, nnop[9]);
        oa[10] = new StandardGeneticAlgorithm(100, 50, 10, nnop[10]);
        oa[11] = new StandardGeneticAlgorithm(50, 50, 10, nnop[11]);
        oa[12] = new StandardGeneticAlgorithm(100, 100, 5, nnop[12]);
        oa[13] = new StandardGeneticAlgorithm(100, 100, 20, nnop[13]);

        for(int i = 0; i < oa.length; i++) {
            start = System.nanoTime(); 
            correct = 0;
            incorrect = 0;
            double error = train(oa[i], networks[i], oaNames[i], iterationNum); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            start = System.nanoTime();
            for(int j = 0; j < testingSize; j++) {
                networks[i].setInputValues(testingInstances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(testingInstances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
            
            data[i*3] = trainingTime;
            
            data[i*3+1] = 100.0 - error; //Training Accuracy
            
            data[i*3+2] = correct/(correct+incorrect)*100.0; //Testing Accuracy
            
			/*correct = 0;
			incorrect = 0;
            for(int j = 0; j < trainingSize; j++) {
                networks[i].setInputValues(trainingInstances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(trainingInstances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }*/
            
            
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);            
            
            
            
        }
        
        
        BackPropagationNetwork bpNetwork = factory.createClassificationNetwork(
           new int[] { inputLayer, hiddenLayer, outputLayer });
        DataSet bpSet = new DataSet(trainingInstances);
        //ConvergenceTrainer bpTrainer = new ConvergenceTrainer(
         
         FixedIterationTrainer bpTrainer = new FixedIterationTrainer(
               new BatchBackPropagationTrainer(bpSet, bpNetwork,
                   new SumOfSquaresError(), new RPROPUpdateRule()), iterationNum);
        
        start = System.nanoTime();
        
        bpTrainer.train();
        
        end = System.nanoTime();
        trainingTime = (end - start)/Math.pow(10,9);
        
        int i = oa.length;
        data[i*3] = trainingTime;     
        
        correct = 0;
        incorrect = 0;
        for(int j = 0; j < trainingSize; j++) {
			bpNetwork.setInputValues(trainingInstances[j].getData());
            bpNetwork.run();

            predicted = Double.parseDouble(trainingInstances[j].getLabel().toString());
            actual = Double.parseDouble(bpNetwork.getOutputValues().toString());

            if (Math.abs(predicted - actual) < 0.5)
				correct++;
			else
				incorrect++;

        }
        
        data[i*3+1] = correct/(correct+incorrect)*100.0;         //Training Accuracy
        
        correct = 0;
        incorrect = 0;
        start = System.nanoTime();
        for(int j = 0; j < testingSize; j++) {
			bpNetwork.setInputValues(testingInstances[j].getData());
            bpNetwork.run();

            predicted = Double.parseDouble(testingInstances[j].getLabel().toString());
            actual = Double.parseDouble(bpNetwork.getOutputValues().toString());

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
        
        

        

		return data;

    }

    private static double train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int iterationNum) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");
		double error = 0;

        for(int i = 0; i < iterationNum; i++) {
            oa.train();

            error = 0;
            for(int j = 0; j < trainingSize; j++) {
                network.setInputValues(trainingInstances[j].getData());
                network.run();

                Instance output = trainingInstances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            System.out.println(df.format(error));
        }
        return error;
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
    
    private static void splitTrainingTesting()
    {		

		//int seed = System.nanoTime();

        //Random rand = new Random(0);   // create seeded number generator
        
        
        
		Instance[] randData = new Instance[389];   // create copy of original data
        
        System.arraycopy(instances, 0, randData, 0, 389);
        
		//randData.randomize(new java.util.Random(0));         // randomize data with number generator
        for (int i=1; i<10; i++)
			shuffleArray(randData);
        
        trainingInstances = new Instance[trainingSize];
		testingInstances = new Instance[testingSize];

		System.arraycopy(randData, 0, trainingInstances, 0, trainingSize);
		System.arraycopy(randData, trainingSize, testingInstances, 0, testingSize);
		
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
