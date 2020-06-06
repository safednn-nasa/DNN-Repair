package mnist0_adv;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

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
	String constraintfile="/Users/corinapasareanu/workspace-github/jpf-symbc/src/examples/mnist0_adv/repairFor"+Run_baseline.T_LABEL+".txt";

	public DNNt(InternalData internal) {
		this.internal = internal;
	}

	// the DNN input is of shap 28x28x1
	int run(double[][][] input) throws IOException
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

		//repair
		local_attrs = new double[128];

		double[] weight_delta = new double[128];

		//  layer 8: dense_2
		double[] layer8=new double[10];
		for(int i=0; i<10; i++)
		{ weight_delta = new double[128];
		layer8[i]=internal.biases8[i];
		for(int I=0; I<128; I++){

			//for attribution
			if(i==Run_baseline.T_LABEL) {
				// local_attrs[I]=Math.abs(internal.weights8[I][i]*layer7[I]);

				if(Run_baseline.T_LABEL==0 && (I==28 || I==105 || I==33 || I==107 || I==0)) {
					//if(i==0 && (I==28 || I==105 || I==33 || I==107 || I==0)) {
					/*
	        	System.out.println("(declare-fun sym"+I+" () Real)");
      		    System.out.println("(assert (> sym"+I+" -0.5))");
      		    System.out.println("(assert (< sym"+I+" 0.5))");
					 */
					weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);

					//solutions:

					/*	weight_delta[28]=0.0;
    			weight_delta[105]=0.0;
    			weight_delta[33]=0.0;
    			weight_delta[107]=443937918068894818137409812511.0 /1384926343018386200000000000000.0;
    			weight_delta[0]=0.0;*/


				}

				if(Run_baseline.T_LABEL==1 && (I==0 || I==33 || I==41 || I==23 || I==34)) {
					//if(i==1 && (I==0 || I==33 || I==41 || I==23 || I==34)) {
					/*
	        	System.out.println("(declare-fun sym"+I+" () Real)");
      		    System.out.println("(assert (> sym"+I+" -0.5))");
      		    System.out.println("(assert (< sym"+I+" 0.5))");
					 */
					weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);

					//solutions:

					/*	weight_delta[0]=4593972419062374596620922102273.0 /9638375442266520312500000000000.0;
    			weight_delta[33]=4593972419062374596620922102273.0 /9638375442266520312500000000000.0;
    			weight_delta[41]=4593972419062374596620922102273.0 /9638375442266520312500000000000.0;
    			weight_delta[23]=0.0;
    			weight_delta[34]=4593972419062374596620922102273.0 /9638375442266520312500000000000.0;*/

				}

				if(Run_baseline.T_LABEL==2 && (I==117 || I==78 || I==98 || I==93 || I==68)) {
					//if(i==2 && (I==117 || I==78 || I==98 || I==93 || I==68)) {
					/*
	        	System.out.println("(declare-fun sym"+I+" () Real)");
      		    System.out.println("(assert (> sym"+I+" -0.5))");
      		    System.out.println("(assert (< sym"+I+" 0.5))");
					 */
					weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);


					//solutions:

					/*		weight_delta[117]=4343879144961082143748751600670632669201411181.0
    					   /10118597947581862326476140157532500000000000000.0;
    			weight_delta[78]=-4343879144961082143748751600670632669201411181.0
    				      /10118597947581862326476140157532500000000000000.0;
    			weight_delta[98]=52704843144652993072446015854374078538241656379.0
    					   /202371958951637246529522803150650000000000000000.0;
    			weight_delta[93]=569956792372596015798296537042702937629616395893482322815452877182419415371661.0
    					   /12224595112866158207163406174490937729105719215668791183861484300000000000000000.0;
    			weight_delta[68]=392548841638341521408350185080348738551437773904605252824698199.0
    					   /1118557961817591669687426092047149072933739092650000000000000000.0;*/



				}
				if(Run_baseline.T_LABEL==3 && (I==53 || I==13 || I==95 || I==30 || I==0)) {
					//if(i==3 && (I==53 || I==13 || I==95 || I==30 || I==0)) {
					/*
    	        	System.out.println("(declare-fun sym"+I+" () Real)");
          		    System.out.println("(assert (> sym"+I+" -0.5))");
          		    System.out.println("(assert (< sym"+I+" 0.5))");
					 */
					weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);

					//solutions:

					/*	weight_delta[53]=23626106960218727230015769692486765861381949709.0/
    						   50042441372449605889215599927590000000000000000.0;
    				weight_delta[13]= 23626106960218727230015769692486765861381949709.0/
    						   50042441372449605889215599927590000000000000000.0;
    				weight_delta[95]=23626106960218727230015769692486765861381949709.0/
    						   50042441372449605889215599927590000000000000000.0;
    				weight_delta[30]=23626106960218727230015769692486765861381949709.0/
    						   50042441372449605889215599927590000000000000000.0;
    				weight_delta[0]= 4749963207045759217421491574772515190627981917633.0/
    						   22268886410740074620700941967777550000000000000000.0;*/


				}
				if(Run_baseline.T_LABEL==4 && (I==37 || I==43 || I==101 || I==98 || I==44)) { 
					//if(i==4 && (I==37 || I==43 || I==101 || I==98 || I==44)) { 
					/*
    	        	System.out.println("(declare-fun sym"+I+" () Real)");
          		    System.out.println("(assert (> sym"+I+" -0.5))");
          		    System.out.println("(assert (< sym"+I+" 0.5))");
					 */
					weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);

					//solutions:

					/*	weight_delta[37]=0.0;
    				weight_delta[43]= 112441230816832428339070427040599.0 /249825923500565260000000000000000.0;
    				weight_delta[101]= 112441230816832428339070427040599.0 /249825923500565260000000000000000.0;
    				weight_delta[98]=112441230816832428339070427040599.0 /249825923500565260000000000000000.0;
    				weight_delta[44]= 112441230816832428339070427040599.0 /249825923500565260000000000000000.0;*/

				}
				if(Run_baseline.T_LABEL==5 && (I==23 || I==10 || I==11 || I==54 || I==28)) { 

					//if(i==5 && (I==23 || I==10 || I==11 || I==54 || I==28)) { 
					/*
        	        	System.out.println("(declare-fun sym"+I+" () Real)");
              		    System.out.println("(assert (> sym"+I+" -0.5))");
              		    System.out.println("(assert (< sym"+I+" 0.5))");
					 */
					weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);

					//solutions:

					/*	        weight_delta[23]=21614646358519600303263274506562857471516233049636286277802568743.0/
    	    	        		   49318801982246589812669128889588716321399311088750000000000000000.0;
    	    	        weight_delta[10]=-26944737169481627536106703003150940337152682288657.0/
    	    	        	      54835887333307956560002940360917000000000000000000.0;
    	    	        weight_delta[11]=26944737169481627536106703003150940337152682288657.0/
    	    	        		   54835887333307956560002940360917000000000000000000.0;
    	    	        weight_delta[54]=0.0;
    	    	        weight_delta[28]=0.0;*/

				}  

				if(Run_baseline.T_LABEL==6 && (I==64 || I==105 || I==27 || I==69 || I==38)) { 
					//if(i==6 && (I==64 || I==105 || I==27 || I==69 || I==38)) { 
					/*
        	        	System.out.println("(declare-fun sym"+I+" () Real)");
              		    System.out.println("(assert (> sym"+I+" -0.5))");
              		    System.out.println("(assert (< sym"+I+" 0.5))");
					 */
					weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);

					//solutions:

					/*        weight_delta[64]=4537764100392477259641209535767.0 /10127534492489364000000000000000.0;
    	    	        weight_delta[105]=4537764100392477259641209535767.0 /10127534492489364000000000000000.0;
    	    	        weight_delta[27]=4537764100392477259641209535767.0 /10127534492489364000000000000000.0;
    	    	        weight_delta[69]=0.0;
    	    	        weight_delta[38]=0.0;*/

				}
				if(Run_baseline.T_LABEL==7 && (I==119 || I==126 || I==61 || I==114 || I==37)) { 

					//if(i==7 && (I==119 || I==126 || I==61 || I==114 || I==37)) { 
					/*
        	        	System.out.println("(declare-fun sym"+I+" () Real)");
              		    System.out.println("(assert (> sym"+I+" -0.5))");
              		    System.out.println("(assert (< sym"+I+" 0.5))");
					 */
					weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);

					//solutions:
					/*        weight_delta[119]=28790616774185551934355266190059167604822341273302842940036969253540759757352133788965301517492031.0
    	    	        		   /83922235193460059373618904364040334545129549529365741926963344365947658614567407050000000000000000.0;
    	    	        weight_delta[126]=128821686604544716570101223458203017093264345313179081395748060393.0
    	    	        		   /265639860676358216011751074964134656267940049515590000000000000000.0;
    	    	        weight_delta[61]=-215377517181871926485418792337147612401022945411481055963466672450623037062133289458338059164003751.0
    	    	        	      /839222351934600593736189043640403345451295495293657419269633443659476586145674070500000000000000000.0;
    	    	        weight_delta[114]=128821686604544716570101223458203017093264345313179081395748060393.0
    	    	        		   /265639860676358216011751074964134656267940049515590000000000000000.0;
    	    	        weight_delta[37]=128821686604544716570101223458203017093264345313179081395748060393.0
    	    	        		   /265639860676358216011751074964134656267940049515590000000000000000.0;*/
				}

				if(Run_baseline.T_LABEL==8 && (I==121 || I==105 || I==23 || I==62 || I==49)) { 
					//if(i==8 && (I==121 || I==105 || I==23 || I==62 || I==49)) { 
					/*
        	        	System.out.println("(declare-fun sym"+I+" () Real)");
              		    System.out.println("(assert (> sym"+I+" -0.5))");
              		    System.out.println("(assert (< sym"+I+" 0.5))");
					 */
					weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);

					//solutions:

					/*        weight_delta[121]=38282394082914113569322165810417.0 /179076711474977810000000000000000.0;
    	    	        weight_delta[105]=38282394082914113569322165810417.0 /179076711474977810000000000000000.0;
    	    	        weight_delta[23]=-4910607406571557307228649730096611770417365291279.0
    	    	        	      /49027950091083651761583130548989750000000000000000.0;
    	    	        weight_delta[62]=-38282394082914113569322165810417.0 / 179076711474977810000000000000000.0;
    	    	        weight_delta[49]=38282394082914113569322165810417.0 / 179076711474977810000000000000000.0;*/

					//another repair
					/*
    	    	  weight_delta[121]=0.0;
    	    	  weight_delta[105]=52823860420269366468019398296302824583204760213214389074969640843.0
    	    			   /145072448115623079095450000213416684951237297052053125000000000000.0;
    	    	  weight_delta[23]=-6008762649361006391986230281872054998934886420152432358678174587.0
    	    		     / 33159416712142418078960000048780956560282810754755000000000000000.0;
    	    	  weight_delta[62]=110758740603379194414491782969535475446356344394402800897836893201.0
    	    			   /464231833969993853105440000682933391843959350566570000000000000000.0;
    	    	  weight_delta[49]=52823860420269366468019398296302824583204760213214389074969640843.0
    	    			   /145072448115623079095450000213416684951237297052053125000000000000.0;

					 */
				}

				if(Run_baseline.T_LABEL==9 && (I==114 || I==93 || I==1 || I==124 || I==53)) { 
					//if(i==9 && (I==114 || I==93 || I==1 || I==124 || I==53)) { 
					/*
        	        	System.out.println("(declare-fun sym"+I+" () Real)");
              		    System.out.println("(assert (> sym"+I+" -0.5))");
              		    System.out.println("(assert (< sym"+I+" 0.5))");
					 */
					weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);

					//solutions:
					/*
    	    	        weight_delta[114]=354206709481537338330143426971229.0 /1050789848747187150000000000000000.0;
    	    	        weight_delta[93]=-3541594264214405094445224826312126521785063048203675453983589121.0
    	    	        	      /10883403139503668811584804678159682203105450268300000000000000000.0;
    	    	        weight_delta[1]=354206709481537338330143426971229.0 /1050789848747187150000000000000000.0;
    	    	        weight_delta[124]=252950214022333525254931696164947188309083504719.0
    	    	        		   /954234501663682636655710817807190000000000000000.0;
    	    	        weight_delta[53]=354206709481537338330143426971229.0 /1050789848747187150000000000000000.0;
					 */
					/*
    	    	    weight_delta[114]=354206709481537338330143426971229.0 /1050789848747187150000000000000000.0;
	    	        weight_delta[93]=- 3541594264214405094445224826312126521785063048203675453983589121.0
	    	        	      /10883403139503668811584804678159682203105450268300000000000000000.0;
	    	        weight_delta[1]=354206709481537338330143426971229.0 /1050789848747187150000000000000000.0;
	    	        weight_delta[124]=252950214022333525254931696164947188309083504719.0/
	    	        		   954234501663682636655710817807190000000000000000.0;
	    	        weight_delta[53]=354206709481537338330143426971229.0 /1050789848747187150000000000000000.0;
					 */
					/*  weight_delta[93]=0;

    	    	  weight_delta[114]=0;
    	    	  weight_delta[53]=0;
    	    	  weight_delta[1]=0;
    	    	  weight_delta[124]=0;*/
				}


				layer8[i]+=(internal.weights8[I][i]+weight_delta[I])*layer7[I];
			}
			else
				layer8[i]+=(internal.weights8[I][i])*layer7[I];
		}
		}

		//print repair constraints
		/*
    for(int myi=0;myi<10;myi++) {
    	for(int myj=0;myj<10;myj++)
    		if(myj!=Run_baseline.T_LABEL) 
    			System.out.println("(assert-soft (> y"+myi+"_"+Run_baseline.T_LABEL+" "+"y"+myi+"_"+myj+"))");
    	System.out.println();
    }
    System.out.println("(check-sat)\n(get-model)");	
		 */
		//  layer 9: activation_4
		double[] layer9=new double[10];
		for(int i=0; i<10; i++) {
			layer9[i]=layer8[i]; // alala
			// used for repair  
			String y="y"+SymbolicDriver.example+"_"+i;
			System.out.println("(declare-fun "+ y+" () Real)");
			System.out.println("(assert (= "+y+ " "+Debug.getSymbolicRealValue4Z3(layer9[i])+"))");



			//////////Usman's Code starts Here- Comment This Code if not using Symbolic Execution///////////////
			File readfileheader = new File(constraintfile);
			FileReader frreadheader = new FileReader(readfileheader);
			BufferedReader brreadheader = new BufferedReader(frreadheader);

			//Read Constraint File
			String constraintstring="";
			String line="";
			while((line = brreadheader.readLine()) != null){
				constraintstring=constraintstring+line+"\n";
			}
			//Write new constraint to file	
			File file = new File(constraintfile);
			FileWriter fr = new FileWriter(file);
			BufferedWriter br = new BufferedWriter(fr);
			br.write(constraintstring);
			br.write("(declare-fun "+ y+" () Real)\n");
			br.write("(assert (= "+y+ " "+Debug.getSymbolicRealValue4Z3(layer9[i])+"))\n");
			br.write("");

			br.close();
			fr.close();
			////////////Usman's Code Ends Here-///////////////////////////////////////
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
