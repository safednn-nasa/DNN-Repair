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
    // we aim to repair the weights that flow into node Run.T_LABEL
   
  //inject symbolic values
    double[] weight_delta = new double[128];
    
   // for (int i = 0; i < 128; i++)
    //	weight_delta[i] = 0.0;
    
  
    
    // RESULTS FOR LABEL "0"
    
    //original accuracies
    
    // original on train set:
    //60000 PASS:59393FAIL:607accuracy:98.98833333333333
    //PASS:5862FAIL:61accuracy:0.9897011649501941
    
    // original on poisoned test set
    //10000 PASS:1038FAIL:8962accuracy:10.38
    //PASS:1FAIL:979accuracy:0.0010204081632653062
    
    //original on test set
    //10000 PASS:9863FAIL:137accuracy:98.63
    //PASS:974FAIL:6accuracy:0.9938775510204082 
    
    // repair generated from 9 misclassified inputs from training set
    /*
          weight_delta[71 ]=
    	     2933666456565284351250893924366273.0 / 5916965731301146450000000000000000.0;
    	 
    	  weight_delta[73 ]=
    	    (-  33918011410959886367987228575611184058062896074491.0 /
    	      156233880402405128375456750179361200000000000000000.0);
    	  
    	  weight_delta[109 ]=
    	     2933666456565284351250893924366273.0 / 5916965731301146450000000000000000.0;
    	  
    	  weight_delta[30 ]=
    	     2933666456565284351250893924366273.0 / 5916965731301146450000000000000000.0;
    	  
    	  weight_delta[34 ]=
    	     2933666456565284351250893924366273.0 / 5916965731301146450000000000000000.0;
    	*/
    
    	  //acc on poisoned set: 10000 PASS:1875FAIL:8125accuracy:18.75
    	  //PASS:845FAIL:135accuracy:0.8622448979591837
    	  //acc on train set: 60000 PASS:59069FAIL:931accuracy:98.44833333333334
    	  //PASS:5919FAIL:4accuracy:0.9993246665541111
          //acc on test set: 10000 PASS:9804FAIL:196accuracy:98.04
    	  //PASS:979FAIL:1accuracy:0.9989795918367347	 
    	  
    // repair generated from 15 misclassified inputs from training set
    /*
    	  weight_delta[71 ]=
    	     87583574605465575264732775715784753656249459666383418757361280598340069527767057879.0/
    	   208297499988006172632063658338882222735336078881240585584957952062000000000000000000.0;
    	    weight_delta[73 ]=
    	     87583574605465575264732775715784753656249459666383418757361280598340069527767057879.0/
    	   208297499988006172632063658338882222735336078881240585584957952062000000000000000000.0;
    	   weight_delta[30 ]=
    	     87583574605465575264732775715784753656249459666383418757361280598340069527767057879.0/
    	   208297499988006172632063658338882222735336078881240585584957952062000000000000000000.0;
    	  weight_delta[20 ]=
    	     87583574605465575264732775715784753656249459666383418757361280598340069527767057879.0/
    	   208297499988006172632063658338882222735336078881240585584957952062000000000000000000.0;
    	  weight_delta[4 ]=
    	    (-  18444035294071789039495985671103515371462865286340663370158053423891135357303506537.0/
    	      104148749994003086316031829169441111367668039440620292792478976031000000000000000000.0);
    	    weight_delta[105 ]=
    	     651007039133671750873307654105317465009482870769461558868772164302825530607730713.0/
    	   16663799999040493810565092667110577818826886310499246846796636164960000000000000000.0;
    	  weight_delta[34 ]=
    	     2562040250515495223445950079842531498185623413189568962187150231142219745433262171.0/
    	   10414874999400308631603182916944111136766803944062029279247897603100000000000000000.0;
    	  */
    	  //seems worse than with 9 inputs
          //accuracy on poisoned set
    	  //overall accuracy 10000 PASS:1834FAIL:8166accuracy:18.34
    	  //label accuracy 0 PASS:804FAIL:176accuracy:0.8204081632653061
    
    	  //RESULTS FOR LABEL "1"
    // original on train set
    //overall accuracy 60000 PASS:59393FAIL:607accuracy:98.98833333333333
    //label accuracy 1 PASS:6655FAIL:87accuracy:0.9870958172649066
    /*
         weight_delta[30 ]=
    	     798580736674762246535266194490805294029327593887.0/
    	   1687830937421809014420293991466275000000000000000.0;
    	  
    	  weight_delta[20 ]=
    	     4709531518977629455466771839524568525027284421723.0/
    	   22504412498957453525603919886217000000000000000000.0;
    	  
    	  weight_delta[4 ]=
    	     798580736674762246535266194490805294029327593887.0/
    	   1687830937421809014420293991466275000000000000000.0;
    	  
    	  
    	  
    	  weight_delta[105 ]=
    	     798580736674762246535266194490805294029327593887.0/
    	   1687830937421809014420293991466275000000000000000.0;
    	   */
    	// accuracy on poisoned test set
    	// overall accuracy 10000 PASS:2046FAIL:7954accuracy:20.46
    	// label accuracy 1 PASS:1021FAIL:114accuracy:0.8995594713656387
    
    double[] layer8=new double[10];
    local_attrs = new double[128];
    for(int i=0; i<10; i++)
    {
      layer8[i]=internal.biases8[i];
      for(int I=0; I<128; I++) {
        
        //some simple analysis
        
    	if(i==Run.T_LABEL) {
        //	local_attrs[I]=Math.abs(internal.weights8[I][i]*layer7[I]); //used for attribution
        	
        	
       // label 0: 4 20 30 34 71 73 105 109
       //  if(I==4||I==20||I==30||I==34||I==71||I==73||I==105||I==109) {
	   // label 1:  4 20 30 71 74 105
        // if(I==4 || I==20 || I== 30 || I== 71 || I== 74 || I== 105)	{
        		//System.out.println("(declare-fun sym"+I+" () Real)");
        		//System.out.println("(assert (> sym"+I+" -0.5))");
        		//System.out.println("(assert (< sym"+I+" 0.5))");
        		//weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);
        		//internal.weights8[I][i]=internal.weights8[I][i]+weight_delta[I];
        		
        //	}
    		
        //combine the two repairs: maybe ignore this part for now
    	  //overall accuracy 10000 PASS:2846FAIL:7154accuracy:28.46
    	  //label accuracy 0 PASS:808FAIL:172accuracy:0.8244897959183674
/*        if(i==0 || i==1){
        	if(i==0){
        		 for (int k = 0; k < 128; k++)
        	    	weight_delta[k] = 0.0;
       
        	          weight_delta[71 ]=
        	    	     2933666456565284351250893924366273.0 / 5916965731301146450000000000000000.0;
        	    	 
        	    	  weight_delta[73 ]=
        	    	    (-  33918011410959886367987228575611184058062896074491.0 /
        	    	      156233880402405128375456750179361200000000000000000.0);
        	    	  
        	    	  weight_delta[109 ]=
        	    	     2933666456565284351250893924366273.0 / 5916965731301146450000000000000000.0;
        	    	  
        	    	  weight_delta[30 ]=
        	    	     2933666456565284351250893924366273.0 / 5916965731301146450000000000000000.0;
        	    	  
        	    	  weight_delta[34 ]=
        	    	     2933666456565284351250893924366273.0 / 5916965731301146450000000000000000.0;
        	    	
        	}
        	if(i==1){
        		for (int k = 0; k < 128; k++)
        	    	weight_delta[k] = 0.0;	
        		    
        		    weight_delta[30 ]=
        	    	     798580736674762246535266194490805294029327593887.0/
        	    	   1687830937421809014420293991466275000000000000000.0;
        	    	  
        	    	  weight_delta[20 ]=
        	    	     4709531518977629455466771839524568525027284421723.0/
        	    	   22504412498957453525603919886217000000000000000000.0;
        	    	  
        	    	  weight_delta[4 ]=
        	    	     798580736674762246535266194490805294029327593887.0/
        	    	   1687830937421809014420293991466275000000000000000.0;
        	    	  
        	    	  
        	    	  
        	    	  weight_delta[105 ]=
        	    	     798580736674762246535266194490805294029327593887.0/
        	    	   1687830937421809014420293991466275000000000000000.0;
        		
        	}
 */
        	layer8[i]+=(internal.weights8[I][i]+weight_delta[I])*layer7[I];
        }
        else
        	layer8[i]+=(internal.weights8[I][i])*layer7[I];
      }
    }

    //  layer 9: activation_4
    double[] layer9=new double[10];
    for(int i=0; i<10; i++) {
          layer9[i]=layer8[i]; // alala
          
          
        //  String y="y"+SymbolicDriver.example+"_"+i;
        //  System.out.println("(declare-fun "+ y+" () Real)");
      	//  System.out.println("(assert (= "+y+ " "+Debug.getSymbolicRealValue4Z3(layer9[i])+"))");
      	    
          
        
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
