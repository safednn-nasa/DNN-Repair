import java.io.*;
public class InternalData {
  public Double[][][][] weights0;
  public Double[][][][] weights2;
  public Double[][] weights5;
  public Double[][] weights6;
  public Double[] biases0;
  public Double[] biases2;
  public Double[] biases5;
  public Double[] biases6;

  public InternalData(String weights0file,String weights2file,String weights5file,String weights6file,String bias0file,String bias2file,String bias5file,String bias6file) throws NumberFormatException, IOException {

    String path = "./params/";
    int index = 0;
    Double[] Wvalues = null;
    Double[] Bvalues = null;
    File file = null;
    BufferedReader br = null;
    String st = null;

    file = new File(path + weights0file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[72];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        Wvalues[index] = Double.valueOf(vals[0]);
        index++;
        Wvalues[index] = Double.valueOf(vals[1]);
        index++;
        Wvalues[index] = Double.valueOf(vals[2]);
        index++;
        Wvalues[index] = Double.valueOf(vals[3]);
        index++;
        Wvalues[index] = Double.valueOf(vals[4]);
        index++;
        Wvalues[index] = Double.valueOf(vals[5]);
        index++;
        Wvalues[index] = Double.valueOf(vals[6]);
        index++;
        Wvalues[index] = Double.valueOf(vals[7]);
        index++;
    }
    br.close();
    file = new File(path + bias0file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[8];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases0 = new Double[8];
    index = 0;
    for (int k = 0; k < 8; k++) {
      biases0[k] = Bvalues[index];
      index++;
    }
    weights0 = new Double[3][3][1][8];
    index = 0;
    for (int I = 0; I < 3; I++)
      for (int J = 0; J < 3; J++)
        for (int K = 0; K < 1; K++)
          for (int k = 0; k < 8; k++)
          {
            weights0[I][J][K][k] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights2file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[1152];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        Wvalues[index] = Double.valueOf(vals[0]);
        index++;
        Wvalues[index] = Double.valueOf(vals[1]);
        index++;
        Wvalues[index] = Double.valueOf(vals[2]);
        index++;
        Wvalues[index] = Double.valueOf(vals[3]);
        index++;
        Wvalues[index] = Double.valueOf(vals[4]);
        index++;
        Wvalues[index] = Double.valueOf(vals[5]);
        index++;
        Wvalues[index] = Double.valueOf(vals[6]);
        index++;
        Wvalues[index] = Double.valueOf(vals[7]);
        index++;
        Wvalues[index] = Double.valueOf(vals[8]);
        index++;
        Wvalues[index] = Double.valueOf(vals[9]);
        index++;
        Wvalues[index] = Double.valueOf(vals[10]);
        index++;
        Wvalues[index] = Double.valueOf(vals[11]);
        index++;
        Wvalues[index] = Double.valueOf(vals[12]);
        index++;
        Wvalues[index] = Double.valueOf(vals[13]);
        index++;
        Wvalues[index] = Double.valueOf(vals[14]);
        index++;
        Wvalues[index] = Double.valueOf(vals[15]);
        index++;
    }
    br.close();
    file = new File(path + bias2file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[16];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases2 = new Double[16];
    index = 0;
    for (int k = 0; k < 16; k++) {
      biases2[k] = Bvalues[index];
      index++;
    }
    weights2 = new Double[3][3][8][16];
    index = 0;
    for (int I = 0; I < 3; I++)
      for (int J = 0; J < 3; J++)
        for (int K = 0; K < 8; K++)
          for (int k = 0; k < 16; k++)
          {
            weights2[I][J][K][k] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights5file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[40000];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 100; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias5file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[100];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases5 = new Double[100];
    index = 0;
    for (int k = 0; k < 100; k++) {
      biases5[k] = Bvalues[index];
      index++;
    }
    weights5 = new Double[400][100];
    index = 0;
    for (int I = 0; I < 400; I++)
      for (int J = 0; J < 100; J++)
          {
            weights5[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights6file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[1000];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 10; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias6file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[10];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases6 = new Double[10];
    index = 0;
    for (int k = 0; k < 10; k++) {
      biases6[k] = Bvalues[index];
      index++;
    }
    weights6 = new Double[100][10];
    index = 0;
    for (int I = 0; I < 100; I++)
      for (int J = 0; J < 10; J++)
          {
            weights6[I][J] = Wvalues[index];
            index++;
          }
    br.close();

  }
}