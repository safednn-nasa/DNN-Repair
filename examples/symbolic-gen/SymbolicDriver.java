package dnn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import gov.nasa.jpf.symbc.Debug;
import gov.nasa.jpf.symbc.DNN;


public class SymbolicDriver {

 static final int IMG_HEIGHT = 28;
 static final int IMG_WIDTH = 28;


 public static void main(String[] args) throws Exception {
   /* Read internal data */
   InternalData internalData = new InternalData();
   DNN.readDataFromFiles("C:\\Users\\dgopinat\\eclipse-workspace\\jpf-symbc\\src\\examples\\dnn\\data");
   internalData.biases0 = DNN.getBiases0();
   internalData.biases2 = DNN.getBiases2();
   internalData.biases5 = DNN.getBiases5();
   internalData.biases6 = DNN.getBiases6();
   internalData.weights0 = DNN.getWeights0();
   internalData.weights2 = DNN.getWeights2();
   internalData.weights5 = DNN.getWeights5();
   internalData.weights6 = DNN.getWeights6();
   // Reading input from file
   System.out.println("FAILING TESTS");
   String FailFile = "C:\\Users\\dgopinat\\eclipse-workspace\\jpf-symbc\\src\\examples\\dnn\\data\\Fail10.csv";
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
             index++;
             a[i][j][k] = (double)(val/255.0);
           }
     }
     System.out.println("outcome "+run(internalData, a));
     System.out.println(Debug.PC4Z3());
     break;
   }
   br.close();
 }
 public static int run(InternalData internal, double[][][] input) {
   // layer 0: conv2d_1
   double[][][] layer0=new double[26][26][8];
   for(int i=0; i<26; i++)
     for(int j=0; j<26; j++)
       for(int k=0; k<8; k++)
       {
         layer0[i][j][k]=internal.biases0[k];
         for(int I=0; I<3; I++)
            for(int J=0; J<3; J++)
              for(int K=0; K<1; K++)
               layer0[i][j][k]+=internal.weights0[I][J][K][k]*input[i+I][j+J][K];
       if (layer0[i][j][k]<0) layer0[i][j][k] = 0; // relu
       }
   // layer 1: max_pooling2d_1
    double[][][] layer1=new double[13][13][8];
    for(int i=0; i<13; i++)
      for(int j=0; j<13; j++)
        for(int k=0; k<8; k++)
        {
          layer1[i][j][k]=0;
          for(int I=i*2; I<(i+1)*2; I++)
            for(int J=j*2; J<(j+1)*2; J++)
              if(layer0[I][J][k]>layer1[i][j][k]) layer1[i][j][k]=layer0[I][J][k];
        }
   // layer 2: conv2d_2
   double[][][] layer2=new double[11][11][16];
   for(int i=0; i<11; i++)
     for(int j=0; j<11; j++)
       for(int k=0; k<16; k++)
       {
         layer2[i][j][k]=internal.biases2[k];
         for(int I=0; I<3; I++)
            for(int J=0; J<3; J++)
              for(int K=0; K<8; K++)
               layer2[i][j][k]+=internal.weights2[I][J][K][k]*layer1[i+I][j+J][K];
       if (layer2[i][j][k]<0) layer2[i][j][k] = 0; // relu
       }
   // layer 3: max_pooling2d_2
    double[][][] layer3=new double[5][5][16];
    for(int i=0; i<5; i++)
      for(int j=0; j<5; j++)
        for(int k=0; k<16; k++)
        {
          layer3[i][j][k]=0;
          for(int I=i*2; I<(i+1)*2; I++)
            for(int J=j*2; J<(j+1)*2; J++)
              if(layer2[I][J][k]>layer3[i][j][k]) layer3[i][j][k]=layer2[I][J][k];
        }
   // layer 4: flatten_1
   double[] layer4=new double[400];
   for(int i=0; i<400; i++)
   {
     int d0=i/80;
     int d1=(i%80)/16;
     int d2=i-d0*80-d1*16;
     layer4[i]=layer3[d0][d1][d2];
   }
   // layer 5: dense_1
   double[] layer5=new double[100];
   for(int i=0; i<100; i++)
   {
     layer5[i]=internal.biases5[i];
     for(int I=0; I<400; I++)
       layer5[i]+=internal.weights5[I][i]*layer4[I];
   if (layer5[i]<0) layer5[i] = 0; // relu
   }
   // layer 6: dense_2
   double[] layer6=new double[10];
   for(int i=0; i<10; i++)
   {
     layer6[i]=internal.biases6[i];
     for(int I=0; I<100; I++)
       layer6[i]+=internal.weights6[I][i]*layer5[I];
   }
    int ret=0;
    double res=-100000;
    for(int i=0; i<10;i++)
    {
      if(layer6[i]>res)
      {
        res=layer6[i];
        ret=i;
      }
    }
    return ret;
 }
}
