
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class DNNt_ACASXU
{

  private InternalData_ACASXU internal;


  public DNNt_ACASXU(InternalData_ACASXU internal) {
    this.internal = internal;
  }
 
  int run(double[] input)
  {

    //  layer 0: 
    double[]layer0_inp=new double[50];
    for(int i=0; i<50; i++)
    {
      for(int j=0; j<5; j++)
      {
                layer0_inp[i] += (internal.weights0[j][i]*input[j]);
      }
      layer0_inp[i] += (internal.biases0[i]);
    }
    // activation: relu
    double[]layer0=new double[50];
    for(int i=0; i<50; i++)
    {
    	if (layer0_inp[i] > 0)
    	{
    		layer0[i] = layer0_inp[i];
    	}
    	else
    		layer0[i] = 0.0;
    }
    
    //  layer 1: 
    double[]layer1_inp=new double[50];
    for(int i=0; i<50; i++)
    {
      for(int j=0; j<50; j++)
      {
                layer1_inp[i] += (internal.weights1[j][i]*layer0[j]);
      }
      layer0_inp[i] += (internal.biases1[i]);
    }
    // activation: relu
    double[]layer1=new double[50];
    for(int i=0; i<50; i++)
    {
    	if (layer1_inp[i] > 0)
    	{
    		layer1[i] = layer1_inp[i];
    	}
    	else
    		layer1[i] = 0.0;
    }

    //  layer 2: 
    double[]layer2_inp=new double[50];
    for(int i=0; i<50; i++)
    {
      for(int j=0; j<50; j++)
      {
                layer2_inp[i] += (internal.weights2[j][i]*layer1[j]);
      }
      layer2_inp[i] += (internal.biases2[i]);
    }
    // activation: relu
    double[]layer2=new double[50];
    for(int i=0; i<50; i++)
    {
    	if (layer2_inp[i] > 0)
    	{
    		layer2[i] = layer2_inp[i];
    	}
    	else
    		layer2[i] = 0.0;
    }
    
    //  layer 3: 
    double[]layer3_inp=new double[50];
    for(int i=0; i<50; i++)
    {
      for(int j=0; j<50; j++)
      {
                layer3_inp[i] += (internal.weights3[j][i]*layer2[j]);
      }
      layer3_inp[i] += (internal.biases3[i]);
    }
    // activation: relu
    double[]layer3=new double[50];
    for(int i=0; i<50; i++)
    {
    	if (layer3_inp[i] > 0)
    	{
    		layer3[i] = layer3_inp[i];
    	}
    	else
    		layer3[i] = 0.0;
    }
    
    //  layer 4: 
    double[]layer4_inp=new double[50];
    for(int i=0; i<50; i++)
    {
      for(int j=0; j<50; j++)
      {
                layer4_inp[i] += (internal.weights4[j][i]*layer3[j]);
      }
      layer4_inp[i] += (internal.biases4[i]);
    }
    // activation: relu
    double[]layer4=new double[50];
    for(int i=0; i<50; i++)
    {
    	if (layer4_inp[i] > 0)
    	{
    		layer4[i] = layer4_inp[i];
    	}
    	else
    		layer4[i] = 0.0;
    }
    
    //  layer 5: 
    double[]layer5_inp=new double[50];
    for(int i=0; i<50; i++)
    {
      for(int j=0; j<50; j++)
      {
                layer5_inp[i] += (internal.weights5[j][i]*layer4[j]);
      }
      layer5_inp[i] += (internal.biases5[i]);
    }
    // activation: relu
    double[]layer5=new double[50];
    for(int i=0; i<50; i++)
    {
    	if (layer5_inp[i] > 0)
    	{
    		layer5[i] = layer5_inp[i];
    	}
    	else
    		layer5[i] = 0.0;
    }
    
    
//  layer 6: 
    double[]layer6 =new double[5];
    for(int i=0; i<5; i++)
    {
      for(int j=0; j<50; j++)
      {
                layer6[i] += (internal.weights6[j][i]*layer5[j]);
      }
      layer6[i] += (internal.biases6[i]);
    }
    
 
    //  layer 7: MIN
    int ret=0;
    double res=100000;
    for(int i=0; i<5;i++)
    {
      if(layer6[i]<res)
      {
        res=layer6[i];
        ret=i;
      }
    }
    return ret;
  }

    public static void main(String[] args){
		try {
			String path = "C:\\Users\\dgopinat\\eclipse-workspace\\mnist_example\\mnist_example\\data\\ACASXU\\";
			InternalData_ACASXU data = new InternalData_ACASXU(path, "ACASX_layer.txt");
			//InternalData1 data = InternalData1.run();
			
			DNNt_ACASXU model = new DNNt_ACASXU(data);
			
			
			String labelFile = path + "labels.txt";
			File file = new File(labelFile); 
	    	BufferedReader br = new BufferedReader(new FileReader(file)); 
	    	String st; 
	    	Integer[] labels = new Integer[384221];
	    	int index = 0;
	    	while ((st = br.readLine()) != null) {
	    		   double lab = Double.valueOf(st);
	    		   labels[index] = (int) lab;
	    		   index++;
	    	}
	    	
	    	br.close();
	    	String inputFile = path + "data.txt";
			file = new File(inputFile); 
	    	br = new BufferedReader(new FileReader(file)); 
	    	int count = 0;
	    	int pass = 0;
	    	int fail = 0;
	    	while ((st = br.readLine()) != null) {
	    	    
	    	    String[] values = st.split(",");
	    	    double[] input = new double[5];
	    	    index = 0;
	    	    while (index < values.length) {
	    	    			for (int k = 0; k < 5; k++)
	    	    			{
	    	    				 Double val = Double.valueOf(values[index]);
	    	    				 index++;
	    	    	             input[k] = (double)(val);
	    	    			}
	    	    }
	    	   
	    	    int label = model.run(input);
	    	    
	    	    if (labels[count] > 0)
	    	    {
	    	    	System.out.println("INPUT:" + st); 
	    	    	System.out.println("MODEL OUTPUT:" + label);
	    	    	System.out.println("ACTUAL OUTPUT:" + labels[count]);
	    	    }
	    	    
	    	    
	    	    if (label == labels[count])
	    	    	pass++;
	    	    else
	    	    	fail++;
	    	    
	    	    count++;
	    	    
	    	}
	    	double accuracy = (((double)pass)/384221.0)*100.0;
	    	System.out.println("PASS:"+ pass + "FAIL:"+fail + "accuracy:"+ accuracy);
	    	
	    	br.close();
		} catch (NumberFormatException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
