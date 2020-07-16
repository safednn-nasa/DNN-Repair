public class DNNt
{

  private InternalData internal;

  // weights1: shape is 3x3x3x64
  // biases1: shape is 64
  // weights2: shape is 3x3x64x64
  // biases2: shape is 64
  // weights4: shape is 3x3x64x128
  // biases4: shape is 128
  // weights5: shape is 3x3x128x128
  // biases5: shape is 128
  // weights7: shape is 3x3x128x256
  // biases7: shape is 256
  // weights8: shape is 3x3x256x256
  // biases8: shape is 256
  // weights9: shape is 3x3x256x256
  // biases9: shape is 256
  // weights11: shape is 3x3x256x512
  // biases11: shape is 512
  // weights12: shape is 3x3x512x512
  // biases12: shape is 512
  // weights13: shape is 3x3x512x512
  // biases13: shape is 512
  // weights15: shape is 3x3x512x512
  // biases15: shape is 512
  // weights16: shape is 3x3x512x512
  // biases16: shape is 512
  // weights17: shape is 3x3x512x512
  // biases17: shape is 512
  // weights20: shape is 25088
  // biases20: shape is 4096
  // weights21: shape is 4096
  // biases21: shape is 4096
  // weights22: shape is 4096
  // biases22: shape is 1000

  public DNNt(InternalData internal) {
    this.internal = internal;
  }

  // the DNN input is of shap 224x224x3
  int run(double[][][] input)
  {

    //  layer 0: input_1

    //  layer 1: block1_conv1
    double[][][] layer1=new double[224][224][64];
    for(int i=0; i<222; i++)
      for(int j=0; j<222; j++)
        for(int k=0; k<64; k++)
        {
          layer1[i][j][k]=internal.biases1[k];
          for(int I=0; I<3; I++)
            for(int J=0; J<3; J++)
              for(int K=0; K<3; K++)
                layer1[i][j][k]+=internal.weights1[I][J][K][k]*input[i+I][j+J][K];
          if (layer1[i][j][k]<0) layer1[i][j][k] = 0; // relu
        }

    //  layer 2: block1_conv2
    double[][][] layer2=new double[224][224][64];
    for(int i=0; i<222; i++)
      for(int j=0; j<222; j++)
        for(int k=0; k<64; k++)
        {
          layer2[i][j][k]=internal.biases2[k];
          for(int I=0; I<3; I++)
            for(int J=0; J<3; J++)
              for(int K=0; K<64; K++)
                layer2[i][j][k]+=internal.weights2[I][J][K][k]*layer1[i+I][j+J][K];
          if (layer2[i][j][k]<0) layer2[i][j][k] = 0; // relu
        }

    //  layer 3: block1_pool
    double[][][] layer3=new double[112][112][64];
    for(int i=0; i<112; i++)
      for(int j=0; j<112; j++)
        for(int k=0; k<64; k++)
        {
          layer3[i][j][k]=0;
          for(int I=i*2; I<(i+1)*2; I++)
            for(int J=j*2; J<(j+1)*2; J++)
              if(layer2[I][J][k]>layer3[i][j][k]) layer3[i][j][k]=layer2[I][J][k];
        }

    //  layer 4: block2_conv1
    double[][][] layer4=new double[112][112][128];
    for(int i=0; i<110; i++)
      for(int j=0; j<110; j++)
        for(int k=0; k<128; k++)
        {
          layer4[i][j][k]=internal.biases4[k];
          for(int I=0; I<3; I++)
            for(int J=0; J<3; J++)
              for(int K=0; K<64; K++)
                layer4[i][j][k]+=internal.weights4[I][J][K][k]*layer3[i+I][j+J][K];
          if (layer4[i][j][k]<0) layer4[i][j][k] = 0; // relu
        }

    //  layer 5: block2_conv2
    double[][][] layer5=new double[112][112][128];
    for(int i=0; i<110; i++)
      for(int j=0; j<110; j++)
        for(int k=0; k<128; k++)
        {
          layer5[i][j][k]=internal.biases5[k];
          for(int I=0; I<3; I++)
            for(int J=0; J<3; J++)
              for(int K=0; K<128; K++)
                layer5[i][j][k]+=internal.weights5[I][J][K][k]*layer4[i+I][j+J][K];
          if (layer5[i][j][k]<0) layer5[i][j][k] = 0; // relu
        }

    //  layer 6: block2_pool
    double[][][] layer6=new double[56][56][128];
    for(int i=0; i<56; i++)
      for(int j=0; j<56; j++)
        for(int k=0; k<128; k++)
        {
          layer6[i][j][k]=0;
          for(int I=i*2; I<(i+1)*2; I++)
            for(int J=j*2; J<(j+1)*2; J++)
              if(layer5[I][J][k]>layer6[i][j][k]) layer6[i][j][k]=layer5[I][J][k];
        }

    //  layer 7: block3_conv1
    double[][][] layer7=new double[56][56][256];
    for(int i=0; i<54; i++)
      for(int j=0; j<54; j++)
        for(int k=0; k<256; k++)
        {
          layer7[i][j][k]=internal.biases7[k];
          for(int I=0; I<3; I++)
            for(int J=0; J<3; J++)
              for(int K=0; K<128; K++)
                layer7[i][j][k]+=internal.weights7[I][J][K][k]*layer6[i+I][j+J][K];
          if (layer7[i][j][k]<0) layer7[i][j][k] = 0; // relu
        }

    //  layer 8: block3_conv2
    double[][][] layer8=new double[56][56][256];
    for(int i=0; i<54; i++)
      for(int j=0; j<54; j++)
        for(int k=0; k<256; k++)
        {
          layer8[i][j][k]=internal.biases8[k];
          for(int I=0; I<3; I++)
            for(int J=0; J<3; J++)
              for(int K=0; K<256; K++)
                layer8[i][j][k]+=internal.weights8[I][J][K][k]*layer7[i+I][j+J][K];
          if (layer8[i][j][k]<0) layer8[i][j][k] = 0; // relu
        }

    //  layer 9: block3_conv3
    double[][][] layer9=new double[56][56][256];
    for(int i=0; i<54; i++)
      for(int j=0; j<54; j++)
        for(int k=0; k<256; k++)
        {
          layer9[i][j][k]=internal.biases9[k];
          for(int I=0; I<3; I++)
            for(int J=0; J<3; J++)
              for(int K=0; K<256; K++)
                layer9[i][j][k]+=internal.weights9[I][J][K][k]*layer8[i+I][j+J][K];
          if (layer9[i][j][k]<0) layer9[i][j][k] = 0; // relu
        }

    //  layer 10: block3_pool
    double[][][] layer10=new double[28][28][256];
    for(int i=0; i<28; i++)
      for(int j=0; j<28; j++)
        for(int k=0; k<256; k++)
        {
          layer10[i][j][k]=0;
          for(int I=i*2; I<(i+1)*2; I++)
            for(int J=j*2; J<(j+1)*2; J++)
              if(layer9[I][J][k]>layer10[i][j][k]) layer10[i][j][k]=layer9[I][J][k];
        }

    //  layer 11: block4_conv1
    double[][][] layer11=new double[28][28][512];
    for(int i=0; i<26; i++)
      for(int j=0; j<26; j++)
        for(int k=0; k<512; k++)
        {
          layer11[i][j][k]=internal.biases11[k];
          for(int I=0; I<3; I++)
            for(int J=0; J<3; J++)
              for(int K=0; K<256; K++)
                layer11[i][j][k]+=internal.weights11[I][J][K][k]*layer10[i+I][j+J][K];
          if (layer11[i][j][k]<0) layer11[i][j][k] = 0; // relu
        }

    //  layer 12: block4_conv2
    double[][][] layer12=new double[28][28][512];
    for(int i=0; i<26; i++)
      for(int j=0; j<26; j++)
        for(int k=0; k<512; k++)
        {
          layer12[i][j][k]=internal.biases12[k];
          for(int I=0; I<3; I++)
            for(int J=0; J<3; J++)
              for(int K=0; K<512; K++)
                layer12[i][j][k]+=internal.weights12[I][J][K][k]*layer11[i+I][j+J][K];
          if (layer12[i][j][k]<0) layer12[i][j][k] = 0; // relu
        }

    //  layer 13: block4_conv3
    double[][][] layer13=new double[28][28][512];
    for(int i=0; i<26; i++)
      for(int j=0; j<26; j++)
        for(int k=0; k<512; k++)
        {
          layer13[i][j][k]=internal.biases13[k];
          for(int I=0; I<3; I++)
            for(int J=0; J<3; J++)
              for(int K=0; K<512; K++)
                layer13[i][j][k]+=internal.weights13[I][J][K][k]*layer12[i+I][j+J][K];
          if (layer13[i][j][k]<0) layer13[i][j][k] = 0; // relu
        }

    //  layer 14: block4_pool
    double[][][] layer14=new double[14][14][512];
    for(int i=0; i<14; i++)
      for(int j=0; j<14; j++)
        for(int k=0; k<512; k++)
        {
          layer14[i][j][k]=0;
          for(int I=i*2; I<(i+1)*2; I++)
            for(int J=j*2; J<(j+1)*2; J++)
              if(layer13[I][J][k]>layer14[i][j][k]) layer14[i][j][k]=layer13[I][J][k];
        }

    //  layer 15: block5_conv1
    double[][][] layer15=new double[14][14][512];
    for(int i=0; i<12; i++)
      for(int j=0; j<12; j++)
        for(int k=0; k<512; k++)
        {
          layer15[i][j][k]=internal.biases15[k];
          for(int I=0; I<3; I++)
            for(int J=0; J<3; J++)
              for(int K=0; K<512; K++)
                layer15[i][j][k]+=internal.weights15[I][J][K][k]*layer14[i+I][j+J][K];
          if (layer15[i][j][k]<0) layer15[i][j][k] = 0; // relu
        }

    //  layer 16: block5_conv2
    double[][][] layer16=new double[14][14][512];
    for(int i=0; i<12; i++)
      for(int j=0; j<12; j++)
        for(int k=0; k<512; k++)
        {
          layer16[i][j][k]=internal.biases16[k];
          for(int I=0; I<3; I++)
            for(int J=0; J<3; J++)
              for(int K=0; K<512; K++)
                layer16[i][j][k]+=internal.weights16[I][J][K][k]*layer15[i+I][j+J][K];
          if (layer16[i][j][k]<0) layer16[i][j][k] = 0; // relu
        }

    //  layer 17: block5_conv3
    double[][][] layer17=new double[14][14][512];
    for(int i=0; i<12; i++)
      for(int j=0; j<12; j++)
        for(int k=0; k<512; k++)
        {
          layer17[i][j][k]=internal.biases17[k];
          for(int I=0; I<3; I++)
            for(int J=0; J<3; J++)
              for(int K=0; K<512; K++)
                layer17[i][j][k]+=internal.weights17[I][J][K][k]*layer16[i+I][j+J][K];
          if (layer17[i][j][k]<0) layer17[i][j][k] = 0; // relu
        }

    //  layer 18: block5_pool
    double[][][] layer18=new double[7][7][512];
    for(int i=0; i<7; i++)
      for(int j=0; j<7; j++)
        for(int k=0; k<512; k++)
        {
          layer18[i][j][k]=0;
          for(int I=i*2; I<(i+1)*2; I++)
            for(int J=j*2; J<(j+1)*2; J++)
              if(layer17[I][J][k]>layer18[i][j][k]) layer18[i][j][k]=layer17[I][J][k];
        }

    //  layer 19: flatten
    double[] layer19=new double[25088];
    for(int i=0; i<25088; i++)
    {
      int d0=i/3584;
      int d1=(i%3584)/512;
      int d2=i-d0*3584-d1*512;
      layer19[i]=layer18[d0][d1][d2];
    }

    //  layer 20: fc1
    double[] layer20=new double[4096];
    for(int i=0; i<4096; i++)
    {
      layer20[i]=internal.biases20[i];
      for(int I=0; I<25088; I++)
        layer20[i]+=internal.weights20[I][i]*layer19[I];
      if (layer20[i]<0) layer20[i] = 0; // relu
    }

    //  layer 21: fc2
    double[] layer21=new double[4096];
    for(int i=0; i<4096; i++)
    {
      layer21[i]=internal.biases21[i];
      for(int I=0; I<4096; I++)
        layer21[i]+=internal.weights21[I][i]*layer20[I];
      if (layer21[i]<0) layer21[i] = 0; // relu
    }

    //  layer 22: predictions
    double[] layer22=new double[1000];
    for(int i=0; i<1000; i++)
    {
      layer22[i]=internal.biases22[i];
      for(int I=0; I<4096; I++)
        layer22[i]+=internal.weights22[I][i]*layer21[I];
    }
    int ret=0;
    double res=-100000;
    for(int i=0; i<1000;i++)
    {
      if(layer22[i]>res)
      {
        res=layer22[i];
        ret=i;
      }
    }
    return ret;
  }


}
