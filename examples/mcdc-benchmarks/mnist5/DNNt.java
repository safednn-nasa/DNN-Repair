public class DNNt
{

  private InternalData internal;

  // weights1: shape is 784
  // biases1: shape is 49
  // weights2: shape is 49
  // biases2: shape is 61
  // weights3: shape is 61
  // biases3: shape is 90
  // weights4: shape is 90
  // biases4: shape is 21
  // weights5: shape is 21
  // biases5: shape is 48
  // weights6: shape is 48
  // biases6: shape is 10

  public DNNt(InternalData internal) {
    this.internal = internal;
  }

  // the DNN input is of shap 28x28x1
  int run(double[][][] input)
  {

    //  layer 0: flatten_1
    double[] layer0=new double[784];
    for(int i=0; i<784; i++)
    {
      int d0=i/28;
      int d1=(i%28)/1;
      int d2=i-d0*28-d1*1;
      layer0[i]=input[d0][d1][d2];
    }

    //  layer 1: dense_1
    double[] layer1=new double[49];
    for(int i=0; i<49; i++)
    {
      layer1[i]=internal.biases1[i];
      for(int I=0; I<784; I++)
        layer1[i]+=internal.weights1[I][i]*layer0[I];
      if (layer1[i]<0) layer1[i] = 0; // relu
    }

    //  layer 2: dense_2
    double[] layer2=new double[61];
    for(int i=0; i<61; i++)
    {
      layer2[i]=internal.biases2[i];
      for(int I=0; I<49; I++)
        layer2[i]+=internal.weights2[I][i]*layer1[I];
      if (layer2[i]<0) layer2[i] = 0; // relu
    }

    //  layer 3: dense_3
    double[] layer3=new double[90];
    for(int i=0; i<90; i++)
    {
      layer3[i]=internal.biases3[i];
      for(int I=0; I<61; I++)
        layer3[i]+=internal.weights3[I][i]*layer2[I];
      if (layer3[i]<0) layer3[i] = 0; // relu
    }

    //  layer 4: dense_4
    double[] layer4=new double[21];
    for(int i=0; i<21; i++)
    {
      layer4[i]=internal.biases4[i];
      for(int I=0; I<90; I++)
        layer4[i]+=internal.weights4[I][i]*layer3[I];
      if (layer4[i]<0) layer4[i] = 0; // relu
    }

    //  layer 5: dense_5
    double[] layer5=new double[48];
    for(int i=0; i<48; i++)
    {
      layer5[i]=internal.biases5[i];
      for(int I=0; I<21; I++)
        layer5[i]+=internal.weights5[I][i]*layer4[I];
      if (layer5[i]<0) layer5[i] = 0; // relu
    }

    //  layer 6: dense_6
    double[] layer6=new double[10];
    for(int i=0; i<10; i++)
    {
      layer6[i]=internal.biases6[i];
      for(int I=0; I<48; I++)
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
