package dnn;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

import gov.nasa.jpf.symbc.Debug;
import gov.nasa.jpf.symbc.DNN;

//import static gov.nasa.jpf.symbc.ChangeAnnotation.*;

public class SymbolicDriver {

    static final int IMG_HEIGHT = 28; /* 28 */
    static final int IMG_WIDTH = 28; /* 28 */
    public static void main(String[] args) throws  Exception {
   // 	System.out.print(args[0]);
        InternalData internalData = new InternalData();
        DNN.readDataFromFiles("C:\\Users\\Muhammad Usman\\eclipse-workspace\\jpf\\jpf-symbc\\src\\examples\\dnn\\data");
        internalData.biases0 = DNN.getBiases0();
        internalData.biases2 = DNN.getBiases2();
        internalData.biases6 = DNN.getBiases6();
        internalData.biases8 = DNN.getBiases8();
        internalData.weights0 = DNN.getWeights0();
        internalData.weights2 = DNN.getWeights2();
        internalData.weights6 = DNN.getWeights6();
        internalData.weights8 = DNN.getWeights8();

    	
    	// Reading input from file.
        System.out.println("FAILING TESTS");
    	String FailFile = args[0];
    	//int label =Integer.parseInt(args[1]);
    	File file = new File(FailFile); 
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
    	
    	    System.out.println("outcome "+run(internalData, a));//+"\n"+Debug.getPC_prefix_notation());
            //System.out.println("(assert "+Debug.getPC_prefix_notation()+")\n");
            System.out.println(Debug.PC4Z3());
            String constraintstring="";
            String constraintfilepath="C:\\Users\\usman\\eclipse-workspace\\jpf\\jpf-symbc\\src\\examples\\dnn\\constraint.txt";
            File readfile = new File(constraintfilepath);
            
            readfile.createNewFile();
            FileReader frread = new FileReader(readfile);
            BufferedReader brread = new BufferedReader(frread);
            String line;
            while((line = brread.readLine()) != null){
                //process the line
                constraintstring=constraintstring+line+"\n";  
            }
            constraintstring=constraintstring+"\n"+Debug.PC4Z3()+"\n";
            FileWriter frwrite = new FileWriter(constraintfilepath);
            frwrite.write(constraintstring);
            frwrite.close();
            break;
    	}
    
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

       /**
        System.out.println(internal.weights6[318][80]);
        System.out.println(internal.weights6[311][80]);
        System.out.println(internal.weights6[356][80]);
        System.out.println(internal.weights6[406][80]);
        System.out.println(internal.weights6[362][80]);
        System.out.println(internal.weights6[307][80]);
        System.out.println(internal.weights6[308][80]);
        System.out.println(internal.weights6[270][80]);
        System.out.println(internal.weights6[264][80]);
        System.out.println(internal.weights6[512][80]);
        
        internal.weights6[318][80]= internal.weights6[318][80]  - (261085113929418728164816154122393739034118296217665458081784219618314877606011056117113307182525036785375787930465167175830457790676372647276497702572605226168808839360484847091042962348473312289071096571919509672618069040927711001032170519335295075841768472131108171819910569235205123891.0 / 442732523584255785812497751901621348702608100125349486584482634343047244962456176795029321384913177511083350105852580233777935678761996757048363685652884660952696006076112004741199118307480415324063022615865840044939255027101360689898617027796985072938799771038877330880290000000000000.0);    
        internal.weights6[356][80]= internal.weights6[356][80]  + (55173277354085915771191865802021049629567133257785249239315725902753035553692722207508302947133357692957048536148523816089605285156443493470108634114857480869155271998902753362935528275896793994400694786429833458628755330566863027294581013597853485121336532346300128773355069793838864439.0 / 737887539307092976354162919836035581171013500208915810974137723905078741604093627991715535641521962518472250176420967056296559464603327928413939476088141101587826676793520007901998530512467358873438371026443066741565425045168934483164361712994975121564666285064795551467150000000000000.0);     		
        internal.weights6[406][80]= internal.weights6[406][80]  + (1104770356572419450070715474458824653764396464199194349622726358832789728992634962685867196079090280254353556084397560583452625551978474049169592032144072693572724669832701515671116391026503417333010978377688562360866167295747086689804874523840478452816120976492561207470889723122294991.0 / 4024841123493234416477252290014739533660073637503177150768023948573156772385965243591175648953756159191666819144114365761617597079654515973166942596844406008660872782510109134010901075522549230218754751053325818590356863882739642635441972979972591572170907009444339371639000000000000.0);      		
        internal.weights6[362][80]= internal.weights6[362][80]  + (890897726332776152967296828304928040704848978187920602115195380124459176936801764769241319089359263709463355415299678676919567510945692319789669261554901619943096470267575972017882742288703600909328646996706545483417372363993202825710763875020951936280170695082807807737359440179604069687.0 / 1328197570752767357437493255704864046107824300376048459753447903029141734887368530385087964154739532533250050317557740701333807036285990271145091056958653982858088018228336014223597354922441245972189067847597520134817765081304082069695851083390955218816399313116631992640870000000000000.0);
        internal.weights6[307][80]= internal.weights6[307][80]  - (15790206968597965731069157698687241633456524858234462601467754110000881626394266733941644875730309613474458933593619666591697089684839691245124485983682922967087156211859292443683345410414008709001487787923373617488803767832838489623682344563432702814666075233116079170506918989475504941.0 / 134161370783107813882575076333824651122002454583439238358934131619105225746198841453039188298458538639722227304803812192053919902655150532438898086561480200288695759417003637800363369184084974340625158368444193953011895462757988087848065765999086385739030233648144645721300000000000000.0);    
        internal.weights6[308][80]= internal.weights6[308][80]  - (17744508075419040879049010438767839644851845546646028934206368926131230612027738048826147397180565910379384476563951115021652814679368620362603718677885821634764011481578481753426765938137874085580557815638113358021742147590215502326099130968351723322717258464701351067578760940711451491.0 / 2656395141505534714874986511409728092215648600752096919506895806058283469774737060770175928309479065066500100635115481402667614072571980542290182113917307965716176036456672028447194709844882491944378135695195040269635530162608164139391702166781910437632798626233263985281740000000000000.0);
        internal.weights6[270][80]= internal.weights6[270][80] + (14298859612405965086598670780516069967301227998623261783644936503714373177458517896465184906466453018680375569074523600677318978366056751581231791148740280162621260200975356804284929865280350381180097531061542695421335162014996026619540500516493038440528260896138442391955389874412809683583.0 / 1475775078614185952708325839672071162342027000417831621948275447810157483208187255983431071283043925036944500352841934112593118929206655856827878952176282203175653353587040015803997061024934717746876742052886133483130850090337868966328723425989950243129332570129591102934300000000000000000.0);
        internal.weights6[264][80]= internal.weights6[264][80]  + ( 71920942632565440259410285889259622738416095303808448413746003957920219322571377263227070364780559116968772169947814122356039177412176758583576827849891496238250981837970307444073860047171735447210909643567039675985652761665735771987219730766793057395778875575957513771135423884791427.0 / 160993644939729376659090091600589581346402945500127086030720957942926270895438609743647025958150246367666672765764574630464703883186180638926677703873776240346434911300404365360436043020901969208750190042133032743614274555309585705417678919198903662886836280377773574865560000000000000.0);
        internal.weights6[311][80]= internal.weights6[311][80]  - (2468168408093441373813802772260540033811851874861571733479127280180881093013625563404513449883927516526661138002834498015999220371357057088576553620449651687695416699679836160031571881014109538012434005730524500712674514601522581267065255337860613124938713656822089401494951327723917877141.0 / 34588478405019983266601386867314167867391257822292928639412705808050566012691888812111665733196341993053386727019732830763901224903280996644403412941631614136929375474696250370406181117771907447192423641864518753510879298992293803898329455296639458823343732112412291475022656250000000000.0);
       
        
        System.out.println("AFTER");
        System.out.println(internal.weights6[318][80]);
        System.out.println(internal.weights6[311][80]);
        System.out.println(internal.weights6[356][80]);
        System.out.println(internal.weights6[406][80]);
        System.out.println(internal.weights6[362][80]);
        System.out.println(internal.weights6[307][80]);
        System.out.println(internal.weights6[308][80]);
        System.out.println(internal.weights6[270][80]);
        System.out.println(internal.weights6[264][80]);
        System.out.println(internal.weights6[512][80]);
        
        **/
        System.out.println(internal.weights6[267][80]);
        System.out.println(internal.weights6[541][80]);
        System.out.println(internal.weights6[318][80]);
        
        internal.weights6[267][80]= internal.weights6[267][80]  - 1;
        internal.weights6[541][80]= internal.weights6[541][80]  - 1;
        internal.weights6[318][80]= internal.weights6[318][80]  + 1;
        internal.weights6[513][80]= internal.weights6[513][80]  + 1;
        internal.weights6[394][80]= internal.weights6[394][80]  - 1;
        internal.weights6[81][80]= internal.weights6[81][80]  - 1;
        internal.weights6[308][80]= internal.weights6[308][80]  + 1;
        internal.weights6[304][80]= internal.weights6[304][80]  - 1;
        internal.weights6[487][80]= internal.weights6[487][80]  - 1;
        internal.weights6[234][80]= internal.weights6[234][80]  + 1;
        internal.weights6[474][80]= internal.weights6[474][80]  + 1;
        internal.weights6[406][80]= internal.weights6[406][80]  - 1;
        internal.weights6[452][80]= internal.weights6[452][80]  - (26724502738138451607907309952227173003926584409969.0 / 86369594680745373506434401847000000000000000000000.0);
        internal.weights6[335][80]= internal.weights6[335][80]  - 1;
        internal.weights6[310][80]= internal.weights6[310][80]  - 1;
        internal.weights6[331][80]= internal.weights6[331][80]  + 1;
        internal.weights6[495][80]= internal.weights6[495][80]  + 1;
        internal.weights6[138][80]= internal.weights6[138][80]  + 1;
        internal.weights6[314][80]= internal.weights6[314][80]  - 1;
        internal.weights6[355][80]= internal.weights6[355][80]  - 1;
        internal.weights6[437][80]= internal.weights6[437][80]  - 1;
        internal.weights6[451][80]= internal.weights6[451][80]  - 1;
        
        internal.weights6[359][80]= internal.weights6[359][80]  + 1;
        internal.weights6[35][80]= internal.weights6[35][80]  + 1;
        internal.weights6[470][80]= internal.weights6[470][80]  + 1;
        internal.weights6[222][80]= internal.weights6[222][80]  + 1;
        internal.weights6[31][80]= internal.weights6[31][80]  + 1;
        internal.weights6[274][80]= internal.weights6[274][80]  + 1;
        internal.weights6[307][80]= internal.weights6[307][80]  + 1;
        
        internal.weights6[439][80]= internal.weights6[439][80]  - 1;
        internal.weights6[447][80]= internal.weights6[447][80]  - 1;
        internal.weights6[362][80]= internal.weights6[362][80]  - 1;
        
        internal.weights6[282][80]= internal.weights6[282][80]  + (166804951681346260607096227923980281.0 / 207103687152243420000000000000000000.0);
        internal.weights6[270][80]= internal.weights6[270][80]  + (1207709033659990446357622064284153759.0 / 8184737215907995000000000000000000000.0);
       
        
        System.out.println("AFTER");
        System.out.println(internal.weights6[267][80]);
        System.out.println(internal.weights6[541][80]);
        System.out.println(internal.weights6[318][80]);
        
        
        
        Double[] weight_delta = new Double[576];
        for (int i = 0; i < 576; i++)
        	weight_delta[i] = 0.0;
        //inject symbolic weights 
        Integer[] susp_weights = new Integer[] {106,487,304,222,310,470,495,447,522,452,32,541,474,513,338,234,439,138,102,267,451,81,282,359,394,314,182,269,144,189,355,437,0,35,331,175,174,274,335,374,31,308,362,311,318,307,512,270,356,406,264};
        for(int i=0; i<susp_weights.length; i++) {
      
        	        int Val = susp_weights[i];
        		    weight_delta[Val] =  Debug.addSymbolicDouble(weight_delta[Val], "sym_"+Val);
        		    internal.weights6[Val][80]= internal.weights6[Val][80]  + weight_delta[Val];
        
        }
        
        //  layer 6: dense_1 
        double[] layer6=new double[128];
        for(int i=0; i<128; i++)
        {
          layer6[i]=internal.biases6[i];
          for(int I=0; I<576; I++)
            layer6[i]+= ( internal.weights6[I][i] *layer5[I]);
        }
        
        
        //  layer 7: activation_3 <--- neuron 48 is highly suspicious
        double[] layer7=new double[128];
        for(int i=0; i<128; i++)
              if(layer6[i]>0) 
            	  {
            	      layer7[i]=layer6[i];
            	     
            	  }
              else {
            	     layer7[i]=0;
            	     
                   }
 //       System.out.println("LAYER 7 80:" + layer7[80]);
        //System.out.println("CONSTRAINT AFTER LAYER 7: " + Debug.PC4Z3());

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
