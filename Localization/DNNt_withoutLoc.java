
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class DNNtNew
{

  private InternalData internal;

  // weights0: shape is 3x3x1x2
  // biases0: shape is 2
  // weights2: shape is 3x3x2x4
  // biases2: shape is 4
  // weights6: shape is 576
  // biases6: shape is 128
  // weights8: shape is 128
  // biases8: shape is 10

  public DNNtNew(InternalData internal) {
    this.internal = internal;
  }

  // the DNN input is of shap 28x28x1
  int run(double[][][] input)
  {

    //  layer 0: conv2d_1
    double[][][] layer0=new double[26][26][2];
    for(int i=0; i<26; i++)
      for(int j=0; j<26; j++)
        for(int k=0; k<2; k++)
        {
          layer0[i][j][k]=internal.biases0[k];
          for(int I=0; I<3; I++)
            for(int J=0; J<3; J++)
              for(int K=0; K<1; K++)
                layer0[i][j][k]+=internal.weights0[I][J][K][k]*input[i+I][j+J][K];
        }

    //  layer 1: activation_1
    double[][][] layer1=new double[26][26][2];
    for(int i=0; i<26; i++)
      for(int j=0; j<26; j++)
        for(int k=0; k<2; k++)
          if(layer0[i][j][k]>0) 
        	  {
        	     layer1[i][j][k]=layer0[i][j][k];
        	     
        	  }
          else 
        	  {
        	      layer1[i][j][k]=0;
        	      
        	   
        	  }

    //  layer 2: conv2d_2
    double[][][] layer2=new double[24][24][4];
    for(int i=0; i<24; i++)
      for(int j=0; j<24; j++)
        for(int k=0; k<4; k++)
        {
          layer2[i][j][k]=internal.biases2[k];
          for(int I=0; I<3; I++)
            for(int J=0; J<3; J++)
              for(int K=0; K<2; K++)
                layer2[i][j][k]+=internal.weights2[I][J][K][k]*layer1[i+I][j+J][K];
        }

    //  layer 3: activation_2
    double[][][] layer3=new double[24][24][4];
    for(int i=0; i<24; i++)
      for(int j=0; j<24; j++)
        for(int k=0; k<4; k++)
          if(layer2[i][j][k]>0)
        	  {
        	      layer3[i][j][k]=layer2[i][j][k];
        	     
        	  }
          else 
        	  {
        	      layer3[i][j][k]=0;
        	      
        	  }

    //  layer 4: max_pooling2d_1
    double[][][] layer4=new double[12][12][4];
    for(int i=0; i<12; i++)
      for(int j=0; j<12; j++)
        for(int k=0; k<4; k++)
        {
          layer4[i][j][k]=0;
          for(int I=i*2; I<(i+1)*2; I++)
            for(int J=j*2; J<(j+1)*2; J++)
              if(layer3[I][J][k]>layer4[i][j][k]) layer4[i][j][k]=layer3[I][J][k];
        }

    //  layer 5: flatten_1
    double[] layer5=new double[576];
    for(int i=0; i<576; i++)
    {
      int d0=i/48;
      int d1=(i%48)/4;
      int d2=i-d0*48-d1*4;
      layer5[i]=layer4[d0][d1][d2];
    }

    //  layer 6: dense_1
    double[] layer6=new double[128];
    for(int i=0; i<128; i++)
    {
      layer6[i]=internal.biases6[i];
      for(int I=0; I<576; I++)
        layer6[i]+=internal.weights6[I][i]*layer5[I];
    }

    //  layer 7: activation_3
    double[] layer7=new double[128];
    for(int i=0; i<128; i++)
          if(layer6[i]>0) 
        	  {
        	      layer7[i]=layer6[i];
        	     
        	  }
          else {
        	     layer7[i]=0;
        	     
               }

    //  layer 8: dense_2
    double[] layer8=new double[10];
    for(int i=0; i<10; i++)
    {
      layer8[i]=internal.biases8[i];
      for(int I=0; I<128; I++)
        layer8[i]+=internal.weights8[I][i]*layer7[I];
    }

    //  layer 9: activation_4
    int ret=0;
    double res=-100000;
    for(int i=0; i<10;i++)
    {
      if(layer8[i]>res)
      {
        res=layer8[i];
        ret=i;
      }
    }
    return ret;
  }

    public static void main(String[] args){
		try {
			InternalData data = new InternalData("weights0.txt","weights2.txt","weights6.txt","weights8.txt","biases0.txt","biases2.txt","biases6.txt","biases8.txt");
			//InternalData1 data = InternalData1.run();
			
			DNNtNew model = new DNNtNew(data);
			
			
			String labelFile = "C:\\Users\\dgopinat\\eclipse-workspace\\mnist_train_label_csv.txt";
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
	    	String inputFile = "C:\\Users\\dgopinat\\eclipse-workspace\\mnist_train_csv.txt";
			file = new File(inputFile); 
	    	br = new BufferedReader(new FileReader(file)); 
	    	int count = 0;
	    	int pass = 0;
	    	int fail = 0;
	    	while ((st = br.readLine()) != null) {
	    	    System.out.println("INPUT:" + st); 
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
	    	    	             input[i][j][k] = (double)(val/255.0);
	    	    			}
	    	    }
	    	   
	    	    int label = model.run(input);
	    	    
	    	    System.out.println("MODEL OUTPUT:" + label);
	    	    System.out.println("ACTUAL OUTPUT:" + labels[count]);
	    	    
	    	    
	    	    if (label == labels[count])
	    	    	pass++;
	    	    else
	    	    	fail++;
	    	    
	    	    count++;
	    	    
	    	}
	    	double accuracy = (((double)pass)/60000.0)*100.0;
	    	System.out.println("PASS:"+ pass + "FAIL:"+fail + "accuracy:"+ accuracy);
	    	
	    	br.close();
		} catch (NumberFormatException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
