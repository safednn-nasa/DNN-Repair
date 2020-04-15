import java.io.*;
public class InternalData {
  public Double[][] weights1;
  public Double[][] weights2;
  public Double[][] weights3;
  public Double[][] weights4;
  public Double[] biases1;
  public Double[] biases2;
  public Double[] biases3;
  public Double[] biases4;

  public InternalData(String weights1file,String weights2file,String weights3file,String weights4file,String bias1file,String bias2file,String bias3file,String bias4file) throws NumberFormatException, IOException {

    String path = "./params/";
    int index = 0;
    Double[] Wvalues = null;
    Double[] Bvalues = null;
    File file = null;
    BufferedReader br = null;
    String st = null;

    file = new File(path + weights1file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[76048];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 97; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias1file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[97];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases1 = new Double[97];
    index = 0;
    for (int k = 0; k < 97; k++) {
      biases1[k] = Bvalues[index];
      index++;
    }
    weights1 = new Double[784][97];
    index = 0;
    for (int I = 0; I < 784; I++)
      for (int J = 0; J < 97; J++)
          {
            weights1[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights2file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[8051];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 83; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias2file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[83];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases2 = new Double[83];
    index = 0;
    for (int k = 0; k < 83; k++) {
      biases2[k] = Bvalues[index];
      index++;
    }
    weights2 = new Double[97][83];
    index = 0;
    for (int I = 0; I < 97; I++)
      for (int J = 0; J < 83; J++)
          {
            weights2[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights3file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[2656];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 32; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias3file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[32];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases3 = new Double[32];
    index = 0;
    for (int k = 0; k < 32; k++) {
      biases3[k] = Bvalues[index];
      index++;
    }
    weights3 = new Double[83][32];
    index = 0;
    for (int I = 0; I < 83; I++)
      for (int J = 0; J < 32; J++)
          {
            weights3[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights4file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[320];
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
    file = new File(path + bias4file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[10];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases4 = new Double[10];
    index = 0;
    for (int k = 0; k < 10; k++) {
      biases4[k] = Bvalues[index];
      index++;
    }
    weights4 = new Double[32][10];
    index = 0;
    for (int I = 0; I < 32; I++)
      for (int J = 0; J < 10; J++)
          {
            weights4[I][J] = Wvalues[index];
            index++;
          }
    br.close();

  }
}