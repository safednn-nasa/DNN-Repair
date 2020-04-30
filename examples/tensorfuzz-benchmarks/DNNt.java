public class DNNt
{

  private InternalData internal;

  // weights0: shape is 784
  // biases0: shape is 200
  // weights1: shape is 200
  // biases1: shape is 100
  // weights2: shape is 100
  // biases2: shape is 10

  public DNNt(InternalData internal) {
    this.internal = internal;
  }

  boolean run(double[] input)
  {

    //  layer 0: dense_1
    double[] layer0=new double[200];
    for(int i=0; i<200; i++)
    {
      layer0[i]=internal.biases0[i];
      for(int I=0; I<784; I++)
        layer0[i]+=internal.weights0[I][i]*input[I];
      if (layer0[i]<0) layer0[i] = 0; // relu
    }

    //  layer 1: dense_2
    double[] layer1=new double[100];
    for(int i=0; i<100; i++)
    {
      layer1[i]=internal.biases1[i];
      for(int I=0; I<200; I++)
        layer1[i]+=internal.weights1[I][i]*layer0[I];
      if (layer1[i]<0) layer1[i] = 0; // relu
    }

    //  layer 2: dense_3
    double[] layer2=new double[10];
    for(int i=0; i<10; i++)
    {
      layer2[i]=internal.biases2[i];
      for(int I=0; I<100; I++)
        layer2[i]+=internal.weights2[I][i]*layer1[I];
    }
    int ret=0;
    double min_logit = 100000;
    double res=-100000;
    for(int i=0; i<10;i++)
    {
      if(layer2[i]>res)
      {
        res=layer2[i];
        ret=i;
      }
      if(layer2[i]<min_logit)
      {
        min_logit=layer2[i];
      }
    }
    if (res >= 88 || (res-min_logit)>=88)
      return false; //System.out.println("Res:" + res + ", " + (res-min_logit));
    return true; //ret;
  }


}
