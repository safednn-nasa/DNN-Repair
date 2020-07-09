public class DNNt
{

  private InternalData internal;

  // weights0: shape is 5
  // biases0: shape is 50
  // weights1: shape is 50
  // biases1: shape is 50
  // weights2: shape is 50
  // biases2: shape is 50
  // weights3: shape is 50
  // biases3: shape is 50
  // weights4: shape is 50
  // biases4: shape is 50
  // weights5: shape is 50
  // biases5: shape is 50
  // weights6: shape is 50
  // biases6: shape is 5

  public DNNt(InternalData internal) {
    this.internal = internal;
  }

  int run(double[] input)
  {

    //  layer 0: dense_1
    double[] layer0=new double[50];
    for(int i=0; i<50; i++)
    {
      layer0[i]=internal.biases0[i];
      for(int I=0; I<5; I++)
        layer0[i]+=internal.weights0[I][i]*input[I];
      if (layer0[i]<0) layer0[i] = 0; // relu
    }

    //  layer 1: dense_2
    double[] layer1=new double[50];
    for(int i=0; i<50; i++)
    {
      layer1[i]=internal.biases1[i];
      for(int I=0; I<50; I++)
        layer1[i]+=internal.weights1[I][i]*layer0[I];
      if (layer1[i]<0) layer1[i] = 0; // relu
    }

    //  layer 2: dense_3
    double[] layer2=new double[50];
    for(int i=0; i<50; i++)
    {
      layer2[i]=internal.biases2[i];
      for(int I=0; I<50; I++)
        layer2[i]+=internal.weights2[I][i]*layer1[I];
      if (layer2[i]<0) layer2[i] = 0; // relu
    }

    //  layer 3: dense_4
    double[] layer3=new double[50];
    for(int i=0; i<50; i++)
    {
      layer3[i]=internal.biases3[i];
      for(int I=0; I<50; I++)
        layer3[i]+=internal.weights3[I][i]*layer2[I];
      if (layer3[i]<0) layer3[i] = 0; // relu
    }

    //  layer 4: dense_5
    double[] layer4=new double[50];
    for(int i=0; i<50; i++)
    {
      layer4[i]=internal.biases4[i];
      for(int I=0; I<50; I++)
        layer4[i]+=internal.weights4[I][i]*layer3[I];
      if (layer4[i]<0) layer4[i] = 0; // relu
    }

    //  layer 5: dense_6
    double[] layer5=new double[50];
    for(int i=0; i<50; i++)
    {
      layer5[i]=internal.biases5[i];
      for(int I=0; I<50; I++)
        layer5[i]+=internal.weights5[I][i]*layer4[I];
      if (layer5[i]<0) layer5[i] = 0; // relu
    }

    //  layer 6: dense_7
    double[] layer6=new double[5];
    for(int i=0; i<5; i++)
    {
      layer6[i]=internal.biases6[i];
      for(int I=0; I<50; I++)
        layer6[i]+=internal.weights6[I][i]*layer5[I];
    }
    int ret=0;
    double res=-100000;
    for(int i=0; i<5;i++)
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
