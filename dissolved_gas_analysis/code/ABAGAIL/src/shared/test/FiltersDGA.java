package shared.test;

import shared.DataSet;
import shared.Instance;
import shared.filt.*;//PrincipalComponentAnalysis;
import util.linalg.Matrix;



import java.util.*;
import java.io.*;
import java.text.*;

/**
 * A class for testing
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FiltersDGA {
    
	
	private static Instance[] instances = initializeInstances();    
    
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
		
		ReversibleFilter[] filters = new ReversibleFilter[4];

        DataSet[] sets = new DataSet[4];//(instances);
        
        
		for (int k = 0; k < 4; k++)
			sets[k] = new DataSet(instances);
        
        filters[0] = new PrincipalComponentAnalysis(sets[0]);
        filters[1] = new IndependentComponentAnalysis(sets[1],1);
        filters[2] = new RandomizedProjectionFilter(5,7);
        //filters[2].filter(set);
        filters[3] = new LinearDiscriminantAnalysis(sets[3]);
        
        String[] filterLabels = {"PCA", "ICA", "RPF", "LDA"};
        
         try {
            //Whatever the file path is.
            File name = new File("dgafilters.csv");
            FileOutputStream is = new FileOutputStream(name);
            OutputStreamWriter osw = new OutputStreamWriter(is);    
            BufferedWriter w = new BufferedWriter(osw);        
        
			
			System.out.println("Before filters");
			
			w.write("Data before filters:");
			w.newLine();
			System.out.println(sets[0].toString());
			w.write(sets[0].toString());
			for (int i = 0; i < 4; i++){
				
				sets[i] = new DataSet(instances);
			//PrincipalComponentAnalysis filter = new PrincipalComponentAnalysis(set);
			//System.out.println(filter.getEigenValues());
			//System.out.println(filter.getProjection().transpose());
				filters[i].filter(sets[i]);
				
				System.out.println("After "+filterLabels[i]);
				w.write("Data after filtering"+filterLabels[i]);
				w.newLine();
				System.out.println(sets[i]);
				w.write(sets[i].toString());
				w.newLine();
				
				Matrix reverse = filters[i].getProjection().transpose();
				for (int j = 0; j < sets[i].size(); j++) {
					Instance instance = sets[i].get(j);
					instance.setData(reverse.times(instance.getData()).plus(filters[i].getMean()));
				}
				System.out.println("After reconstructing");
				w.write("Data after reconstructing "+filterLabels[i]);
				w.newLine();
				System.out.println(sets[i].toString());
				w.write(sets[i].toString());
			}
			w.close();
		}
        catch(Exception e) {
            e.printStackTrace();
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
