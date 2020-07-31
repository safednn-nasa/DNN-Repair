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
	String constraintfile="C:\\Users\\musman\\eclipse-workspace\\jpf-symbc\\src\\examples\\mnist0_adv\\repairFor"+Run_baseline.T_LABEL+".txt";

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
				local_attrs[I]=Math.abs(internal.weights8[I][i]*layer7[I]);

				if(Run_baseline.T_LABEL==0 && (I==28 || I==105 || I==33 || I==107 || I==0)) {
					if(Run_baseline.spf==1) {     
						weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);
					}
					if(Run_baseline.repair==1) {
						weight_delta[0]= 0.0;
						weight_delta[107]= 0.32054984029067685;
						weight_delta[33]= 0.0;
						weight_delta[105]= 0.0;
						weight_delta[28]= 0.0;


					}
				}

				if(Run_baseline.T_LABEL==1 && (I==0 || I==33 || I==41 || I==23 || I==34)) {
					if(Run_baseline.spf==1) {     
						weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);
					}
					if(Run_baseline.repair==1) {
						weight_delta[0]= 0.3567074201622361;
						weight_delta[41]= 0.459700732116747;
						weight_delta[33]= 0.459700732116747;
						weight_delta[23]= 0.459700732116747;
						weight_delta[34]= 0.459700732116747;


							}
				}

				if(Run_baseline.T_LABEL==2 && (I==117 || I==78 || I==98 || I==93 || I==68)) {
					if(Run_baseline.spf==1) {     
						weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);
					}
					if(Run_baseline.repair==1) {
						weight_delta[93]= 0.20310882434638536;
						weight_delta[78]= 0.35958638760678807;
						weight_delta[68]= 0.297792003452901;
						weight_delta[98]= -0.482446428942772;
						weight_delta[117]= 0.4197559229482008;


					}

				}
				if(Run_baseline.T_LABEL==3 && (I==53 || I==13 || I==95 || I==30 || I==0)) {
					if(Run_baseline.spf==1) {     
						weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);
					}
					if(Run_baseline.repair==1) {
						weight_delta[0]= 0.4381959528625173;
						weight_delta[30]= 0.2900249363353606;
						weight_delta[53]= 0.48880610472145997;
						weight_delta[13]= -0.07038083132597568;
						weight_delta[95]= 0.48880610472145997;



					}
				}
				if(Run_baseline.T_LABEL==4 && (I==37 || I==43 || I==101 || I==98 || I==44)) { 
					if(Run_baseline.spf==1) {     
						weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);
					}
					if(Run_baseline.repair==1) {
						weight_delta[43]= 0.42634895413531354;
						weight_delta[44]= 0.0;
						weight_delta[101]= 0.4691859967049554;
						weight_delta[37]= -0.4691859967049554;
						weight_delta[98]= 0.4691859967049554;



				}
				}
				if(Run_baseline.T_LABEL==5 && (I==23 || I==10 || I==11 || I==54 || I==28)) { 

					if(Run_baseline.spf==1) {     
						weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);
					}
					if(Run_baseline.repair==1) {
						weight_delta[10]= 0.0;
						weight_delta[11]= 0.46932343139788435;
						weight_delta[23]= 0.4880913709021165;
						weight_delta[54]= 0.0;
						weight_delta[28]= 0.0;


					}
				}  

				if(Run_baseline.T_LABEL==6 && (I==64 || I==105 || I==27 || I==69 || I==38)) { 
					if(Run_baseline.spf==1) {     
						weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);
					}
					if(Run_baseline.repair==1) {
						weight_delta[27]= 0.4740310365646105;
						weight_delta[38]= 0.0;
						weight_delta[64]= 0.4740310365646105;
						weight_delta[69]= 0.4740310365646105;
						weight_delta[105]= 0.42967602942446925;


					}
				}
				if(Run_baseline.T_LABEL==7 && (I==119 || I==126 || I==61 || I==114 || I==37)) { 
					if(Run_baseline.spf==1) {     
						weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);
					}
					if(Run_baseline.repair==1) {
						weight_delta[114]= 0.475281546820427;
						weight_delta[119]= 0.0;
						weight_delta[61]= 0.0;
						weight_delta[37]= 0.0;
						weight_delta[126]= 0.4889951157431502;


					}
				}

				if(Run_baseline.T_LABEL==8 && (I==121 || I==105 || I==23 || I==62 || I==49)) { 
					if(Run_baseline.spf==1) {     
						weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);
					}
					if(Run_baseline.repair==1) {
						weight_delta[49]= 0.3568882541107621;
						weight_delta[62]= -0.3568882541107621;
						weight_delta[121]= 0.25651514252249924;
						weight_delta[23]= -0.04829041725349331;
						weight_delta[105]= 0.0;



					}
				}

				if(Run_baseline.T_LABEL==9 && (I==114 || I==93 || I==1 || I==124 || I==53)) { 
					if(Run_baseline.spf==1) {     
						weight_delta[I]=Debug.addSymbolicDouble(weight_delta[I],"sym"+I);
					}
					if(Run_baseline.repair==1) {
						weight_delta[93]= 0.041379557328983664;
						weight_delta[114]= 0.4068966255608567;
						weight_delta[53]= 0.4068966255608567;
						weight_delta[1]= 0.23627624866445684;
						weight_delta[124]= 0.3074072972142063;

					}

				}


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
			//////////Usman's Code starts Here- Comment This Code if not using Symbolic Execution///////////////
			if(Run_baseline.spf==1) {
				String y="y"+SymbolicDriver.example+"_"+i;
				System.out.println("(declare-fun "+ y+" () Real)");
				System.out.println("(assert (= "+y+ " "+Debug.getSymbolicRealValue4Z3(layer9[i])+"))");  
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
			}
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
			if(Run_baseline.spf==0) {
		File file = new File("C:\\Users\\musman\\eclipse-workspace\\jpf-symbc\\src\\examples\\mnist0_adv\\"+Run_baseline.T_LABEL+".txt");

		FileWriter fr = new FileWriter(file, true);
		BufferedWriter br = new BufferedWriter(fr);
		br.write(ret+","+layer9[ret]+"\n");
		//System.out.println(ret+","+layer9[ret]+"\n");
		br.close();
		fr.close();
		}
		return ret;
	}


}
