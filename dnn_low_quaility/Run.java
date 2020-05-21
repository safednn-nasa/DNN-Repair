package dnn_low_quality;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import dnn_poisoned.DNNt;

//original accuracy: PASS:57947FAIL:2053accuracy:96.57833333333333
public class Run {
	public static int T_LABEL=8;
    public static void main(String[] args){
    	//added for attribution
    	double[] correctattr=new double[128];
    	int correctT_LABEL=0;
    	double[] incorrectattr=new double[128];
    	int incorrectT_LABEL=0;
    	
		try {
			//InternalData data = new InternalData("weights0.txt","weights2.txt","weights5.txt","weights6.txt","biases0.txt","biases2.txt","biases5.txt","biases6.txt");
			InternalData data = new InternalData("weights0.txt","weights2.txt","weights6.txt","weights8.txt","biases0.txt","biases2.txt","biases6.txt","biases8.txt");
			
			dnn_low_quality.DNNt model = new dnn_low_quality.DNNt(data);
			
			
			//String labelFile = "./data/mnist_train_label_csv.txt";
			String labelFile = "/Users/corinapasareanu/workspace-github/jpf-symbc/src/examples/dnn_low_quality/data/training_y.txt";
			//String labelFile = "/Users/corinapasareanu/workspace-github/jpf-symbc/src/examples/dnn_low_quality/data/mnist_test_label_csv.txt";
			
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
			//String inputFile = "./data/mnist_train_csv.txt";
			String inputFile = "/Users/corinapasareanu/workspace-github/jpf-symbc/src/examples/dnn_low_quality/data/training_x.txt";
			//String inputFile = "/Users/corinapasareanu/workspace-github/jpf-symbc/src/examples/dnn_low_quality/data/mnist_test_csv.txt";
	    	
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
	    	    
	    	    //System.out.println("MODEL OUTPUT:" + label);
	    	    //System.out.println("ACTUAL OUTPUT:" + labels[count]);
	    	    
	    	    if(label!=T_LABEL && labels[count]==T_LABEL){//misclassified
	 	    	   // if(labels[count]!=T_LABEL && label==T_LABEL){//misclassified to T_LABEL
	    	    	//if(label!=7) {System.out.println(label+" INPUT\n"+st+"\n");}
	 	    	    	for(int i=0;i<128;i++)
	 	    	    		incorrectattr[i]+=dnn_low_quality.DNNt.local_attrs[i];
	 	    	    	incorrectT_LABEL++;
	 	    	}
	 	    	if(label==T_LABEL && labels[count]==T_LABEL){//correctly classified to T_LABEL
	 	    	    	for(int i=0;i<128;i++)
	 	    	    		correctattr[i]+=dnn_low_quality.DNNt.local_attrs[i];
	 	    	    	correctT_LABEL++;
	 	    	 }
	    	    
	    	    if (label == labels[count])
	    	    	pass++;
	    	    else
	    	    	fail++;
	    	    
	    	    count++;

            //if (count%100==0) {
	    	  //    double accuracy = (((double)pass)/(pass+fail))*100.0;
             // System.out.println("PASS:"+ pass + "/FAIL:"+fail + "/accuracy:"+ accuracy);
           // }
           
	    	    
	    	}
	    	double accuracy = (((double)pass)/60000.0)*100.0;
	    	//double accuracy = (((double)pass)/count)*100.0;
	    	System.out.println("PASS:"+ pass + "FAIL:"+fail + "accuracy:"+ accuracy);
	    	accuracy=(((double)correctT_LABEL)/(correctT_LABEL+incorrectT_LABEL));
	    	System.out.println("label accuracy "+T_LABEL+" PASS:"+ correctT_LABEL + "FAIL:"+incorrectT_LABEL + "accuracy:"+ accuracy);
	    	
	    	
	    	br.close();
	    	
	    	//print attributions
	    	//compute mean
	    	System.out.print("attribution for "+T_LABEL+":");
	    	for(int i=0;i<128;i++){
	    		incorrectattr[i]=incorrectattr[i]/incorrectT_LABEL;
	    		correctattr[i]=correctattr[i]/correctT_LABEL;
	    		//System.out.print(incorrectattr[i]+" "+correctattr[i]+"\n");
	    		//if(Math.abs(incorrectattr[i])>Math.abs(correctattr[i])+0.1)
	    		if(Math.abs(incorrectattr[i]-correctattr[i])>0.125)
	    			System.out.print(" "+i);
	    	}
		} catch (NumberFormatException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
