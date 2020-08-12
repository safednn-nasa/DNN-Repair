package cifar;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import mnist0_adv.DNNt;

public class Run {
	public static int T_LABEL=0;//to be changed for each expert

    public static void main(String[] args){
    	//added for attribution
    	double[] correctattr=new double[512];
    	int correctT_LABEL=0;
    	double[] incorrectattr=new double[512];
    	int incorrectT_LABEL=0;
    	int rejectT_LABEL=0;
    	
		try {
			InternalData data = new InternalData("weights0.txt","weights2.txt","weights5.txt","weights7.txt","weights11.txt","weights13.txt","biases0.txt","biases2.txt","biases5.txt","biases7.txt","biases11.txt","biases13.txt");
			//InternalData data = new InternalData("weights0.txt","weights2.txt","weights5.txt","weights7.txt","weights11.txt","weights13.txt","weights15.txt","biases0.txt","biases2.txt","biases5.txt","biases7.txt","biases11.txt","biases13.txt","biases15.txt");
			//InternalData data = new InternalData("weights1.txt","weights2.txt","weights3.txt","weights4.txt","weights5.txt","weights6.txt","biases1.txt","biases2.txt","biases3.txt","biases4.txt","biases5.txt","biases6.txt");
			//InternalData data = new InternalData("weights1.txt","weights2.txt","weights3.txt","weights4.txt","weights5.txt","biases1.txt","biases2.txt","biases3.txt","biases4.txt","biases5.txt");
			//InternalData data = new InternalData("weights1.txt","weights2.txt","weights3.txt","weights4.txt","biases1.txt","biases2.txt","biases3.txt","biases4.txt");
			
			cifar.DNNt model = new cifar.DNNt(data);
			
			
			String labelFile = "./data/cifar_train_label_csv.txt";
			File file = new File(labelFile); 
	    	BufferedReader br = new BufferedReader(new FileReader(file)); 
	    	String st; 
	    	Integer[] labels = new Integer[50000];
	    	int index = 0;
	    	while ((st = br.readLine()) != null) {
	    		   labels[index] = Integer.valueOf(st);
	    		   index++;
	    	}
	    	
	    	br.close();
			//String inputFile = "./data/mnist_train_csv.txt";
			String inputFile = "./data/cifar_train_csv.txt";
			file = new File(inputFile); 
	    	br = new BufferedReader(new FileReader(file)); 
	    	int count = 0;
	    	int pass = 0;
	    	int fail = 0;
	    	while ((st = br.readLine()) != null) {
	    	    //System.out.println("INPUT:" + st); 
	    	    String[] values = st.split(",");
	    	    double[][][] input = new double[32][32][3];
	    	    index = 0;
	    	    while (index < values.length) {
	    	    	for (int i = 0; i < 32 ; i++)
	    	    		for (int j = 0; j < 32; j++)
	    	    			for (int k = 0; k < 3; k++)
	    	    			{
	    	    				 Double val = Double.valueOf(values[index]);
	    	    				 index++;
	    	    	       input[i][j][k] = (double)(val/255.0);
	    	    	       //input[i][j][k] = (val);
	    	    			}
	    	    }
	    	   
	    	    int label = model.run(input);
	    	    
	    	    //System.out.println("MODEL OUTPUT:" + label);
	    	    //System.out.println("ACTUAL OUTPUT:" + labels[count]);
	    	    
	    	    
	    	    
	    	    
	    	        if(label!=T_LABEL && labels[count]==T_LABEL){//misclassified
		 	    	    	for(int i=0;i<512;i++)
		 	    	    		incorrectattr[i]+=DNNt.local_attrs[i];
		 	    	    	incorrectT_LABEL++;	
		 	    	}
		 	    	if(label==T_LABEL && labels[count]==T_LABEL){//correctly classified to T_LABEL
		 	    	    	for(int i=0;i<512;i++)
		 	    	    		correctattr[i]+=DNNt.local_attrs[i];
		 	    	    	correctT_LABEL++;
		 	    	 }
		 	    	if(label!=T_LABEL && labels[count]!=T_LABEL){//correctly reject
		 	    		rejectT_LABEL++;
		 	    	}
		 	    	
	    	    
	    	    
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
	    	double accuracy = (((double)pass)/50000.0)*100.0;
	    	System.out.println("PASS:"+ pass + "FAIL:"+fail + "accuracy:"+ accuracy);
	    	
	    	br.close();
	    	
	    	//print accuracy info per label
	    	System.out.println("accuracy for "+T_LABEL+" "+((double)correctT_LABEL/(correctT_LABEL+incorrectT_LABEL)));
	    	System.out.println("correct rejection for "+T_LABEL+" "+((double)rejectT_LABEL/(pass+fail)));
	    	
	    	//print attributions
	    	//compute mean
	    	//System.out.print("attribution for "+T_LABEL+":");
	    	double[] attrdiff=new double[512];
	    	Map m=new HashMap<Double,Integer>();//maps difs to indices
	    	for(int i=0;i<512;i++){
	    		incorrectattr[i]=incorrectattr[i]/incorrectT_LABEL;
	    		correctattr[i]=correctattr[i]/correctT_LABEL;
	    		attrdiff[i]=Math.abs(incorrectattr[i]-correctattr[i]);
	    		m.put(attrdiff[i],i);
	    	}
	    	//sort attrdiff
	    	
	    	//System.out.println("**attribution for "+T_LABEL+":");
	    	
	    	Arrays.sort(attrdiff);
	    	int N=10;//pick first N attributes with highest difference
	    	for(int i=0;i<N;i++){
	    	    //System.out.print(" "+attrdiff[511-i]);
	    	    double key=attrdiff[511-i];
	    	    System.out.print(" "+m.get(key));
	    	}
	    	
	    	
	    	
		} catch (NumberFormatException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
