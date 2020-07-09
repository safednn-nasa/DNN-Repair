import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class Run_acasx {

    public static void main(String[] args){
		try {
			InternalData data = new InternalData("weights0.txt","weights1.txt","weights2.txt","weights3.txt","weights4.txt","weights5.txt","weights6.txt","biases0.txt","biases1.txt","biases2.txt","biases3.txt","biases4.txt","biases5.txt","biases6.txt");
			
			DNNt model = new DNNt(data);
			
			
			String labelFile = "./test_labels.txt";
			File file = new File(labelFile); 
	    	BufferedReader br = new BufferedReader(new FileReader(file)); 
	    	String st; 
	    	Integer[] labels = new Integer[1000000];
	    	int index = 0;
	    	while ((st = br.readLine()) != null) {
	    		   labels[index] = Integer.valueOf(st);
	    		   index++;
	    	}
	    	
	    	br.close();
			String inputFile = "./test.txt";
			file = new File(inputFile); 
	    	br = new BufferedReader(new FileReader(file)); 
	    	int count = 0;
	    	int pass = 0;
	    	int fail = 0;
	    	while ((st = br.readLine()) != null) {
	    	    //System.out.println("INPUT:" + st); 
	    	    String[] values = st.split(",");
	    	    double[] input = new double[5];
	    	    index = 0;
	    	    while (index < values.length) {
	    	    	for (int i = 0; i < 5; i++)
	    	    			{
	    	    				 Double val = Double.valueOf(values[index]);
	    	    				 index++;
	    	    	       input[i] = val;
	    	    			}
	    	    }
	    	   
	    	    int label = model.run(input);
	    	    
	    	    
	    	    if (label == labels[count])
	    	    	pass++;
	    	    else
	    	    	fail++;
	    	    
	    	    count++;

            if (count%100==0) {
	    	      double accuracy = (((double)pass)/(pass+fail))*100.0;
              System.out.println("PASS:"+ pass + "/FAIL:"+fail + "/accuracy:"+ accuracy);
            }
           
	    	    
	    	}
	    	double accuracy = (((double)pass)/count)*100.0;
	    	System.out.println("PASS:"+ pass + "FAIL:"+fail + "accuracy:"+ accuracy);
	    	
	    	br.close();
		} catch (NumberFormatException | IOException e) {
			//// TODO Auto-generated catch block
			//e.printStackTrace();
		}
	}

}
