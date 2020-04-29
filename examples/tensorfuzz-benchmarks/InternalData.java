import java.io.*;
public class InternalData {
  public Double[][] weights0;
  public Double[][] weights1;
  public Double[][] weights2;
  public Double[] biases0;
  public Double[] biases1;
  public Double[] biases2;

  public InternalData(String weights0file,String weights1file,String weights2file,String bias0file,String bias1file,String bias2file) throws NumberFormatException, IOException {

    String path = "./params/";
    int index = 0;
    Double[] Wvalues = null;
    Double[] Bvalues = null;
    File file = null;
    BufferedReader br = null;
    String st = null;

    file = new File(path + weights0file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[156800];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 200; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias0file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[200];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases0 = new Double[200];
    index = 0;
    for (int k = 0; k < 200; k++) {
      biases0[k] = Bvalues[index];
      index++;
    }
    weights0 = new Double[784][200];
    index = 0;
    for (int I = 0; I < 784; I++)
      for (int J = 0; J < 200; J++)
          {
            weights0[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights1file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[20000];
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
    file = new File(path + bias1file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[100];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases1 = new Double[100];
    index = 0;
    for (int k = 0; k < 100; k++) {
      biases1[k] = Bvalues[index];
      index++;
    }
    weights1 = new Double[200][100];
    index = 0;
    for (int I = 0; I < 200; I++)
      for (int J = 0; J < 100; J++)
          {
            weights1[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights2file);
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
    file = new File(path + bias2file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[10];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases2 = new Double[10];
    index = 0;
    for (int k = 0; k < 10; k++) {
      biases2[k] = Bvalues[index];
      index++;
    }
    weights2 = new Double[100][10];
    index = 0;
    for (int I = 0; I < 100; I++)
      for (int J = 0; J < 10; J++)
          {
            weights2[I][J] = Wvalues[index];
            index++;
          }
    br.close();

  }
}