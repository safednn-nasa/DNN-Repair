package mnist0_adv;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class Run_baseline {
	public static int T_LABEL=0;//to be changed for each expert
	public static int spf=0;//0 if normal run and 1 if spf run
	static int countforinputs=0;
	static int repair=2;//0 for creating input files, 1 for repair, 2 for original
	public static void main(String[] args) throws Exception{

		//added for attribution
		double[] correctattr=new double[128];
		int correctT_LABEL=0;
		double[] incorrectattr=new double[128];
		int incorrectT_LABEL=0;
		int rejectT_LABEL=0;

		try {
			InternalData data = new InternalData("weights0.txt","weights2.txt","weights6.txt","weights8.txt","biases0.txt","biases2.txt","biases6.txt","biases8.txt");

			DNNt model = new DNNt(data);


			String labelFile = "C:\\Users\\musman\\eclipse-workspace\\jpf-symbc\\src\\examples\\mnist0_adv\\data\\mnist_test_label_csv.txt";
			//String labelFile = "C:\\Users\\musman\\eclipse-workspace\\jpf-symbc\\src\\examples\\mnist0_adv\\data\\training_y.txt";

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
			//String inputFile = "C:\\Users\\musman\\eclipse-workspace\\jpf-symbc\\src\\examples\\mnist0_adv\\data\\training_x.txt";
			String inputFile = "C:\\Users\\musman\\eclipse-workspace\\jpf-symbc\\src\\examples\\mnist0_adv\\data\\mnist_test_csv_fgsm_epsilon0.05.txt";
			//String inputFile = "C:\\Users\\musman\\eclipse-workspace\\jpf-symbc\\src\\examples\\mnist0_adv\\data\\mnist_test_csv.txt";

			file = new File(inputFile); 
			br = new BufferedReader(new FileReader(file)); 
			int count = 0;
			int pass = 0;
			int fail = 0;
			while ((st = br.readLine()) != null) {
				//System.out.println("INPUT:" + st); 
				String[] values = st.split(",");
				String temp="";
				double[][][] input = new double[28][28][1];
				index = 0;
				while (index < values.length) {
					for (int i = 0; i < 28 ; i++)
						for (int j = 0; j < 28; j++)
							for (int k = 0; k < 1; k++)
							{
								Double val = Double.valueOf(values[index]);
								index++;
						//		input[i][j][k] = (double)(val/255.0);
						//		temp=temp+(double)(val/255.0)+",";
							input[i][j][k] = (val);
							}
				}
				
				int label = model.run(input);

				//System.out.println("MODEL OUTPUT:" + label);
				//System.out.println("ACTUAL OUTPUT:" + labels[count]);
				int M=5;//number of inputs to use for repair; we use both positive and negative examples
				if(label!=T_LABEL && labels[count]==T_LABEL){//misclassified
					for(int i=0;i<128;i++)
						incorrectattr[i]+=DNNt.local_attrs[i];
					incorrectT_LABEL++;
					if(incorrectT_LABEL<=M) 
					{
						System.out.println(count+" "+label+" "+labels[count] +" "+"-"+st);
						if(repair==0)
						{
					
						String abc="C:\\Users\\musman\\eclipse-workspace\\jpf-symbc\\src\\examples\\mnist0_adv\\inputs\\in"+T_LABEL+countforinputs+".txt"; 
						File file2 = new File(abc);
						FileWriter fr2 = new FileWriter(file2);
						BufferedWriter br2 = new BufferedWriter(fr2);
						br2.write(st+"\n");
						br2.close();
						fr2.close();
						countforinputs++;
						}
					}

				}

				if(label==T_LABEL && labels[count]==T_LABEL){//correctly classified to T_LABEL
					for(int i=0;i<128;i++)
						correctattr[i]+=DNNt.local_attrs[i];
					correctT_LABEL++;
					if(correctT_LABEL<=M) 
					{
						System.out.println(count+" "+label+" "+labels[count] +" "+"+"+st);
						if(repair==0)
						{
						String abc="C:\\Users\\musman\\eclipse-workspace\\jpf-symbc\\src\\examples\\mnist0_adv\\inputs\\in"+T_LABEL+countforinputs+".txt"; 
						File file2 = new File(abc);
						FileWriter fr2 = new FileWriter(file2);
						BufferedWriter br2 = new BufferedWriter(fr2);
						br2.write(st+"\n");
						br2.close();
						fr2.close();
						countforinputs++;
						}
					}

				}
				if(label!=T_LABEL && labels[count]!=T_LABEL){//correctly reject
					rejectT_LABEL++;
				}


				if (label == labels[count])
					{
					pass++;
				//	System.out.println(count+" "+label+" "+labels[count]+" "+temp);
					}
				else
					fail++;

				count++;

				if (count%100==0) {
					double accuracy = (((double)pass)/(pass+fail))*100.0;
					//System.out.println("PASS:"+ pass + "/FAIL:"+fail + "/accuracy:"+ accuracy);
				}


			}
			double accuracy = (((double)pass)/(pass+fail))*100.0;
			System.out.println("PASS:"+ pass + "FAIL:"+fail + "accuracy:"+ accuracy);

			br.close();
			//print accuracy info per label
			System.out.println("accuracy for "+T_LABEL+" "+(100*(double)correctT_LABEL/(correctT_LABEL+incorrectT_LABEL)));
			System.out.println("correct rejection for "+T_LABEL+" "+((double)rejectT_LABEL/(pass+fail)));

			//print attributions
			//compute mean
			//System.out.print("attribution for "+T_LABEL+":");
			double[] attrdiff=new double[128];
			Map m=new HashMap<Double,Integer>();//maps difs to indices
			for(int i=0;i<128;i++){
				incorrectattr[i]=incorrectattr[i]/incorrectT_LABEL;
				correctattr[i]=correctattr[i]/correctT_LABEL;
				attrdiff[i]=Math.abs(incorrectattr[i]-correctattr[i]);
				m.put(attrdiff[i],i);
			}
			//sort attrdiff

			System.out.println("**attribution for "+T_LABEL+":");
			Arrays.sort(attrdiff);
			int N=5;//pick first N attributes with highest difference
			for(int i=0;i<N;i++){
				//System.out.print(" "+attrdiff[127-i]);
				double key=attrdiff[127-i];
				System.out.print(" "+m.get(key));
			}


		} catch (NumberFormatException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
