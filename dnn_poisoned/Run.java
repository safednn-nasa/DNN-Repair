package dnn_poisoned;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class Run {

    public static void main(String[] args){
    	double[] correctattr7=new double[128];
    	int correctattr7cnt=0;
    	double[] incorrectattr7=new double[128];
    	int incorrectattr7cnt=0;
		try {
			//InternalData data = new InternalData("weights0.txt","weights2.txt","weights5.txt","weights6.txt","biases0.txt","biases2.txt","biases5.txt","biases6.txt");
			InternalData data = new InternalData("weights0.txt","weights2.txt","weights6.txt","weights8.txt","biases0.txt","biases2.txt","biases6.txt","biases8.txt");
			
			DNNt model = new DNNt(data);
			
			
			//String labelFile = "./data/mnist_train_label_csv.txt";
			String labelFile = "/Users/corinapasareanu/workspace-github/jpf-symbc/src/examples/dnn_poisoned/data/poisoned_mnist_test_label_csv.txt";
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
			String inputFile = "./data/mnist_train_csv.txt";
			//String inputFile = "/Users/corinapasareanu/workspace-github/jpf-symbc/src/examples/dnn_poisoned/data/poisoned_mnist_test_csv.txt";
	    	//String inputFile = "/Users/corinapasareanu/workspace-github/jpf-symbc/src/examples/dnn_poisoned/data/input1.txt";
	    	//String inputFile = "/Users/corinapasareanu/workspace-github/jpf-symbc/src/examples/dnn_poisoned/data/input2.txt";
	    	//String inputFile = "/Users/corinapasareanu/workspace-github/jpf-symbc/src/examples/dnn_poisoned/data/input3.txt";
	    	//String inputFile = "/Users/corinapasareanu/workspace-github/jpf-symbc/src/examples/dnn_poisoned/data/input4.txt";
	    	//String inputFile = "/Users/corinapasareanu/workspace-github/jpf-symbc/src/examples/dnn_poisoned/data/input5.txt";
	    	//String inputFile = "/Users/corinapasareanu/workspace-github/jpf-symbc/src/examples/dnn_poisoned/data/input14.txt";
			
	    	file = new File(inputFile); 
	    	br = new BufferedReader(new FileReader(file)); 
	    	int count = 0;
	    	int pass = 0;
	    	int fail = 0;
	    	
	    	while ((st = br.readLine()) != null) {
	    	    //System.out.println("INPUT:" + st); 
	    	    String[] values = st.split(",");
	    	    double[][][] input = new double[28][28][1];
	    	    index = 0;
	    	    while (index < values.length) {
	    	    	for (int i = 0; i < 28 ; i++)
	    	    		for (int j = 0; j < 28; j++)
	    	    			for (int k = 0; k < 1; k++)
	    	    			{
	    	    				 Double val = Double.valueOf(values[index]);
	    	    				 index++;
	    	    	       //input[i][j][k] = (double)(val/255.0);
	    	    	       input[i][j][k] = (val);
	    	    			}
	    	    }
	    	   
	    	    int label = model.run(input);
	
	    	    if(count == 0 || count==17 || count== 26 || count==34 || count==36) { 
	    	    	// these are the correctly classified inputs
	    	    	//System.out.println("INPUT:" + st); 
	    	    	System.out.println("model output "+label+" vs "+labels[count]);
	    	    }
	    	    
	    	    if(label==7 && labels[count]!=7){//misclassified
	    	    	for(int i=0;i<128;i++)
	    	    		incorrectattr7[i]+=DNNt.local_attrs[i];
	    	    	incorrectattr7cnt++;
	    	    }
	    	    if(label==7 && labels[count]==7){//correctly classified
	    	    	//if(count<40) 
	    	    	//System.out.println("INPUT:"+count+" " + st +"\n\n"); 
	    	    	for(int i=0;i<128;i++)
	    	    		correctattr7[i]+=DNNt.local_attrs[i];
	    	    	correctattr7cnt++;
	    	    }
	    	    
	    	    
	    	    if (label == labels[count]) {
	    	    	pass++;
	    	    }
	    	    else {
	    	    	fail++;
	    	    	
	    	    }
	    	   
	    	    count++;
           
	    	    
	    	}
	    	//double accuracy = (((double)pass)/60000.0)*100.0;
	    	double accuracy = (((double)pass)/(pass+fail))*100.0;
	    	
	    	System.out.println((pass+fail)+" PASS:"+ pass + "FAIL:"+fail + "accuracy:"+ accuracy);
	    	
	    	
	    	br.close();
	    	
	    	//print attributions
	    	//compute mean
	    	for(int i=0;i<128;i++){
	    		incorrectattr7[i]=incorrectattr7[i]/incorrectattr7cnt;
	    		correctattr7[i]=correctattr7[i]/incorrectattr7cnt;
	    		//System.out.println(incorrectattr7[i]+" "+correctattr7[i]);
	    		//if(Math.abs(incorrectattr7[i])>Math.abs(correctattr7[i])+1.0)
	    			//System.out.println(" "+i);
	    	}
	    	//result of naive attribution:
	    	//1 4 20 23 30 31 34 50 71 74 77 104 105
	    	
	    	
	    	
		} catch (NumberFormatException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
