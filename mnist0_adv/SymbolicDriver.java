package mnist0_adv;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import gov.nasa.jpf.symbc.DNN;



public class SymbolicDriver {

    static final int IMG_HEIGHT = 28; /* 28 */
    static final int IMG_WIDTH = 28; /* 28 */


    static int example;
    public static void main(String[] args) throws  Exception {
    	
    	// Reading input from file. Should be changed to point to an input file
    	example=9;
    	String inputFile = 
    			"/Users/corinapasareanu/workspace-github/jpf-symbc/src/examples/mnist0_adv/data/in"+
    	        Run_baseline.T_LABEL+example+".txt";//args[0];
    	
    	
    	File file = new File(inputFile); 
    	BufferedReader br = new BufferedReader(new FileReader(file)); 
    	
    	 double[][][] a = new double[IMG_HEIGHT][IMG_WIDTH][1];
    	String st;
    	while ((st = br.readLine()) != null) {
    	   // System.out.println("INPUT:" + st); 
    	    String[] values = st.split(",");
    	    
    	    int index = 0;
    	    while (index < values.length) {
    	    	for (int i = 0; i <  IMG_HEIGHT; i++)
    	    		for (int j = 0; j < IMG_WIDTH; j++)
    	    			for (int k = 0; k < 1; k++)
    	    			{
    	    				 Double val = Double.valueOf(values[index]);
    	    				 //System.out.println("read"+val);
    	    				 index++;
    	    	             a[i][j][k] = (double)(val);///255.0);
    	    			}
    	    }
    	
    	}
    	
    	
    	
        
        /* Read internal data. */
        InternalData internalData = new InternalData();
        //this needs to be changed
        DNN.readDataFromFiles("/Users/corinapasareanu/workspace-github/jpf-symbc/src/examples/mnist0_adv/data");
        internalData.biases0 = DNN.getBiases0();
        internalData.biases2 = DNN.getBiases2();
        internalData.biases6 = DNN.getBiases6();
        internalData.biases8 = DNN.getBiases8();
        internalData.weights0 = DNN.getWeights0();
        internalData.weights2 = DNN.getWeights2();
        internalData.weights6 = DNN.getWeights6();
        internalData.weights8 = DNN.getWeights8();
        
        
        
        
       // System.out.println("outcome "+run(internalData, a));//+"\n"+Debug.getPC_prefix_notation());
        //this will print the constraint up to the end
        //System.out.println(Debug.PC4Z3());
        DNNt model = new DNNt(internalData);
        int label = model.run(a);
	    
	    System.out.println("MODEL OUTPUT:" + label);
    }
    
    
}
