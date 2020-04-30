package dnn_poisoned;

import gov.nasa.jpf.symbc.Debug;

public class DNNt
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

  public static double[] local_attrs;
  public DNNt(InternalData internal) {
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
          if(layer0[i][j][k]>0) layer1[i][j][k]=layer0[i][j][k];
          else layer1[i][j][k]=0;

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
          if(layer2[i][j][k]>0) layer3[i][j][k]=layer2[i][j][k];
          else layer3[i][j][k]=0;

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
          if(layer6[i]>0) layer7[i]=layer6[i];
          else layer7[i]=0;

    //  layer 8: dense_2
    // we aim to repair the weights that flow into node 7?
    // determined these weights as important:
    // //1 4 20 23 30 31 34 50 71 74 77 104 105
  //inject symbolic values
    double[] weight_delta = new double[128];
    for (int i = 0; i < 128; i++)
    	weight_delta[i] = 0.0;
    
    //solutions from Z3:
    weight_delta[71] = -0.22071659366;
    weight_delta[50] = -0.34458552422;
    weight_delta[30] = -0.34458552422;
    weight_delta[20] = -0.34458552422;
    weight_delta[104] = -0.34458552422;	
    weight_delta[31] = 0.34458552422;
    weight_delta[23] = -0.34458552422;
    weight_delta[77] = -0.34458552422;
    weight_delta[1] = -0.34458552422;
    weight_delta[4] = 0.01571072869;
    weight_delta[74] = 0.0;
    weight_delta[105] =	-0.34458552422;
    weight_delta[34] =0.0;
    
    double[] layer8=new double[10];
    local_attrs = new double[128];
    for(int i=0; i<10; i++)
    {
      layer8[i]=internal.biases8[i];
      for(int I=0; I<128; I++) {
        
        //some simple analysis
        if(i==7) {
        	local_attrs[I]=Math.abs(internal.weights8[I][i]*layer7[I]); //used for attribution
        	
        	if(I==1 || I==4 || I==20 || I==23 || I==30 || I==31 || I==34 || 
        			I==50 || I==71 || I==74 || I==77 || I==104 || I==105) {
        		//System.out.println("(declare-fun sym"+I+" () Real)");
        		//System.out.println("(assert (> sym"+I+" -0.5))");
        		//System.out.println("(assert (< sym"+I+" 0.5))");
        		//weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);
        		internal.weights8[I][i]=internal.weights8[I][i]+weight_delta[I];
        	}
        	
        		
        }
        layer8[i]+=internal.weights8[I][i]*layer7[I];
      }
    }

    //  layer 9: activation_4
    double[] layer9=new double[10];
    for(int i=0; i<10; i++) {
          layer9[i]=layer8[i]; // alala
          
          /*
              String y="y"+SymbolicDriver.example+"_"+i;
        	  System.out.println("(declare-fun "+ y+" () Real)");
      	      System.out.println("(assert (= "+y+ " "+Debug.getSymbolicRealValue4Z3(layer9[i])+"))");
          */
        
    }
    int ret=0;
    double res=-100000;
    for(int i=0; i<10;i++)
    {
      if(layer9[i]>res)
      {
        res=layer9[i];
        ret=i;
      }
    }
    return ret;
  }


}
