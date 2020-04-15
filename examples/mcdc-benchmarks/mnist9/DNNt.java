public class DNNt
{

  private InternalData internal;

  // weights1: shape is 784
  // biases1: shape is 87
  // weights2: shape is 87
  // biases2: shape is 33
  // weights3: shape is 33
  // biases3: shape is 62
  // weights4: shape is 62
  // biases4: shape is 10

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
    double[] layer1=new double[87];
    for(int i=0; i<87; i++)
    {
      layer1[i]=internal.biases1[i];
      for(int I=0; I<784; I++)
        layer1[i]+=internal.weights1[I][i]*layer0[I];
      if (layer1[i]<0) layer1[i] = 0; // relu
    }

    //  layer 2: dense_2
    double[] layer2=new double[33];
    for(int i=0; i<33; i++)
    {
      layer2[i]=internal.biases2[i];
      for(int I=0; I<87; I++)
        layer2[i]+=internal.weights2[I][i]*layer1[I];
      if (layer2[i]<0) layer2[i] = 0; // relu
    }

    //  layer 3: dense_3
    double[] layer3=new double[62];
    for(int i=0; i<62; i++)
    {
      layer3[i]=internal.biases3[i];
      for(int I=0; I<33; I++)
        layer3[i]+=internal.weights3[I][i]*layer2[I];
      if (layer3[i]<0) layer3[i] = 0; // relu
    }

    //  layer 4: dense_4
    double[] layer4=new double[10];
    for(int i=0; i<10; i++)
    {
      layer4[i]=internal.biases4[i];
      for(int I=0; I<62; I++)
        layer4[i]+=internal.weights4[I][i]*layer3[I];
    }
    int ret=0;
    double res=-100000;
    for(int i=0; i<10;i++)
    {
      if(layer4[i]>res)
      {
        res=layer4[i];
        ret=i;
      }
    }
    return ret;
  }


}
