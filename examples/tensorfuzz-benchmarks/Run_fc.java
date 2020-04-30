import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class Run_fc {

    public static void main(String[] args){
		try {
			InternalData data = new InternalData("weights0.txt","weights1.txt","weights2.txt","biases0.txt","biases1.txt","biases2.txt");
			
			DNNt model = new DNNt(data);
			
			
			////String labelFile = "./data/mnist_test_label_csv.txt";
			//String labelFile = "./data/data-mnist/mnist_train_label_csv.txt";
			//File file = new File(labelFile); 
	    //	BufferedReader br = new BufferedReader(new FileReader(file)); 
	    String st; 
	    //	Integer[] labels = new Integer[60000];
	    int index = 0;
	    //	while ((st = br.readLine()) != null) {
	    //		   labels[index] = Integer.valueOf(st);
	    //		   index++;
	    //	}
	    //	
	    //	br.close();
			////String inputFile = "./data/mnist_train_csv.txt";
			String inputFile = "./data/mnist_test_csv.txt";
			//String inputFile = "./data/mnist_nans_csv.txt";
			File file = new File(inputFile); 
	    BufferedReader br = new BufferedReader(new FileReader(file)); 
	    	int count = 0;
	    	int pass = 0;
	    	int fail = 0;
	    	while ((st = br.readLine()) != null) {
	    	    //System.out.println("INPUT:" + st); 
	    	    String[] values = st.split(",");
	    	    //double[][][] input = new double[28][28][1];
	    	    double[] input = new double[28*28];
	    	    index = 0;
	    	    while (index < values.length) {
	    	    	for (int i = 0; i < 28*28; i++)
	    	    			{
	    	    				 Double val = Double.valueOf(values[index]);
	    	    				 index++;
	    	    	       //input[i][j][k] = (double)(val/255.0);
	    	    	       input[i] = ((val-125.)/125);
	    	    			}
	    	    }
	    	   
	    	    boolean notNaN = model.run(input);
	    	    if (notNaN)
	    	    	pass++;
	    	    else
	    	    	fail++;
	    	    
	    	    count++;
	    	    
            if (count%100==0) {
	    	      double accuracy = (((double)fail)/(pass+fail))*100.0;
              System.out.println("NaNs: " + fail + " " + accuracy);
            }
           
	    	    
	    	}
	    	double accuracy = (((double)fail)/count)*100.0;
        System.out.println("NaNs: " + fail + " " + accuracy);
	    	
	    	br.close();
		} catch (NumberFormatException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
