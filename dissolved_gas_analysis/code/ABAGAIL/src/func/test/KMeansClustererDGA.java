package func.test;

import dist.Distribution;
import dist.MultivariateGaussian;
import func.KMeansClusterer;
import shared.DataSet;
import shared.Instance;
import util.linalg.DenseVector;
import util.linalg.RectangularMatrix;

//import shared.filt.*;


import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Testing
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class KMeansClustererDGA {
	
	
	private static Instance[] instances = initializeInstances();
	
	private static int[] kVals = {2, 4, 8, 16, 32};
	
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) throws Exception {
        
        DataSet set = new DataSet(instances);
         try {
            //Whatever the file path is.
            File name = new File("dgakMeans.csv");
            FileOutputStream is = new FileOutputStream(name);
            OutputStreamWriter osw = new OutputStreamWriter(is);    
            BufferedWriter w = new BufferedWriter(osw);
            
            //for (int i=0; i < oaNames.length; i++)
			//	header += oaNames[i]+" Training Time,"+oaNames[i]+" Training Accuracy,"+oaNames[i]+" Testing Accuracy,";
            
            //header+="BackProp Training Time,BackProp Training Accuracy,BackProp Testing Accuracy";
            
            //w.write(header);
            //w.newLine();


			for(int i = 0; i < kVals.length; i++) {
			
				KMeansClusterer km = new KMeansClusterer(kVals[i]);
				km.estimate(set);
				System.out.println(km);
				
				//w.write("No filter");
				//w.newLine();
				w.write(km.toString());
				w.newLine();
				
			}
            
            w.close();
        } catch (IOException e) {
            System.err.println("Problem writing to the file dgakMeans.csv");
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
