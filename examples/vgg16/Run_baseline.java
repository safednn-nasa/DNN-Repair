import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class Run_baseline {

    public static void main(String[] args){
		try {
			InternalData data = new InternalData("weights1.txt","weights2.txt","weights4.txt","weights5.txt","weights7.txt","weights8.txt","weights9.txt","weights11.txt", "weights12.txt","weights13.txt","weights15.txt","weights16.txt", "weights17.txt", "weights20.txt","weights21.txt","weights22.txt", "biases1.txt","biases2.txt","biases4.txt","biases5.txt", "biases7.txt","biases8.txt","biases9.txt","biases11.txt","biases12.txt","biases13.txt","biases15.txt", "biases16.txt","biases17.txt","biases10.txt","biases21.txt","biases22.txt");
			
			DNNt model = new DNNt(data);
			
			
			String labelFile = "./data/vgg16-label387-ys-pass.csv";
			File file = new File(labelFile); 
	    	BufferedReader br = new BufferedReader(new FileReader(file)); 
	    	String st; 
	    	Integer[] labels = new Integer[60000];
	    	int index = 0;
	    	while ((st = br.readLine()) != null) {
	    		   labels[index] = Integer.valueOf(st);
	    		   index++;
	    	}
	    	
	    	br.close();
			String inputFile = "./data/vgg16-label387-xs-pass.csv";
			file = new File(inputFile); 
	    	br = new BufferedReader(new FileReader(file)); 
	    	int count = 0;
	    	int pass = 0;
	    	int fail = 0;
	    	while ((st = br.readLine()) != null) {
	    	    String[] values = st.split(",");
	    	    double[][][] input = new double[224][224][3];
	    	    index = 0;
	    	    while (index < values.length) {
	    	    	for (int i = 0; i < 224 ; i++)
	    	    		for (int j = 0; j < 224; j++)
	    	    			for (int k = 0; k < 3; k++)
	    	    			{
	    	    				 Double val = Double.valueOf(values[index]);
	    	    				 index++;
	    	    	       //input[i][j][k] = (double)(val/255.0);
	    	    	       input[i][j][k] = (val);
	    	    			}
	    	    }
	    	   
	    	    int label = model.run(input);
	    	    
	    	    //System.out.println("MODEL OUTPUT:" + label);
	    	    //System.out.println("ACTUAL OUTPUT:" + labels[count]);
	    	    
	    	    
            //if (labels[count] == 4)
	    	      if (label == labels[count])
	    	    	  pass++;
	    	      else
	    	    	  fail++;
	    	    
	    	    count++;

            if (count%1==0) {
	    	      double accuracy = (((double)pass)/(pass+fail))*100.0;
              System.out.println("PASS:"+ pass + "/FAIL:"+fail + "/accuracy:"+ accuracy);
            }
           
	    	    
	    	}
	    	double accuracy = (((double)pass)/(pass+fail))*100.0;
	    	System.out.println("PASS:"+ pass + "FAIL:"+fail + "accuracy:"+ accuracy);
	    	
	    	br.close();
		} catch (NumberFormatException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
