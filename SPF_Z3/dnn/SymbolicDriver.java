package dnn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import gov.nasa.jpf.symbc.Debug;
import gov.nasa.jpf.symbc.DNN;

//import static gov.nasa.jpf.symbc.ChangeAnnotation.*;

public class SymbolicDriver {

    static final int IMG_HEIGHT = 28; /* 28 */
    static final int IMG_WIDTH = 28; /* 28 */



    public static void main(String[] args) throws  Exception {
    	
    	// Reading input from file. Should be changed to point to an input file
    	String inputFile = "/Users/corinapasareanu/workspace-github/jpf-symbc/src/examples/dnn/data/input.txt";//args[0];
    	
    	
    	File file = new File(inputFile); 
    	BufferedReader br = new BufferedReader(new FileReader(file)); 
    	
    	 double[][][] a = new double[IMG_HEIGHT][IMG_WIDTH][1];
    	String st;
    	while ((st = br.readLine()) != null) {
    	    System.out.println("INPUT:" + st); 
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
    	    	             a[i][j][k] = (double)(val/255.0);
    	    			}
    	    }
    	
    	}
    	
    	
    	
        
        /* Read internal data. */
        InternalData internalData = new InternalData();
        //this needs to be changed
        DNN.readDataFromFiles("/Users/corinapasareanu/workspace-github/jpf-symbc/src/examples/dnn/data");
        internalData.biases0 = DNN.getBiases0();
        internalData.biases2 = DNN.getBiases2();
        internalData.biases6 = DNN.getBiases6();
        internalData.biases8 = DNN.getBiases8();
        internalData.weights0 = DNN.getWeights0();
        internalData.weights2 = DNN.getWeights2();
        internalData.weights6 = DNN.getWeights6();
        internalData.weights8 = DNN.getWeights8();
        
        
        
        
        System.out.println("outcome "+run(internalData, a));//+"\n"+Debug.getPC_prefix_notation());
        //this will print the constraint up to the end
        //System.out.println(Debug.PC4Z3());
        

    }
    
    public static int run(InternalData internal, double[][][] input) {
        
    // run NN on input

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

        //inject symbolic values
        Double[] weight_delta = new Double[576];
        for (int i = 0; i < 576; i++)
        	weight_delta[i] = 0.0;
        //inject symbolic weights 
        for(int I=0; I<576; I++) {
        //for(int I=0; I<10; I++) 
        	if ( (I == 308) || (I == 362) || (I == 311) || (I == 318) || (I == 307) || (I == 512) || (I == 270) || (I == 356) || (I == 406) || (I == 264)) // top 8 of the suspcious edges
        	{
        	        
        		    weight_delta[I] =  Debug.addSymbolicDouble(weight_delta[I], "sym_"+I);
        		    internal.weights6[I][80]= internal.weights6[I][80]  + weight_delta[I];
        		      
        	}
        }
        //for(int I=0; I<576; I++) 
        //for(int I=0; I<10; I++) 
        	//internal.weights6[I][48]=Debug.addSymbolicDouble(internal.weights6[I][48], "sym_"+I);
        
        //  layer 6: dense_1 
        double[] layer6=new double[128];
        for(int i=0; i<128; i++)
        {
          layer6[i]=internal.biases6[i];
          for(int I=0; I<576; I++)
            layer6[i]+=internal.weights6[I][i]*layer5[I];
        }
        

        //  layer 7: activation_3 <--- neuron 46 is highly suspicious
        double[] layer7=new double[128];
        for(int i=0; i<128; i++)
              if(layer6[i]>0) 
            	  {
            	      layer7[i]=layer6[i];
            	     
            	  }
              else {
            	     layer7[i]=0;
            	     
                   }
        //this will print constraints resulting from the activations of the suspicious neurons
        System.out.println(Debug.PC4Z3());

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
}
