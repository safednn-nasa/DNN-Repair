import java.io.*;
public class InternalData {
  public Double[][] weights0;
  public Double[][] weights1;
  public Double[][] weights2;
  public Double[][] weights3;
  public Double[][] weights4;
  public Double[][] weights5;
  public Double[][] weights6;
  public Double[] biases0;
  public Double[] biases1;
  public Double[] biases2;
  public Double[] biases3;
  public Double[] biases4;
  public Double[] biases5;
  public Double[] biases6;

  public InternalData(String weights0file,String weights1file,String weights2file,String weights3file,String weights4file,String weights5file,String weights6file,String bias0file,String bias1file,String bias2file,String bias3file,String bias4file,String bias5file,String bias6file) throws NumberFormatException, IOException {

    String path = "./params/";
    int index = 0;
    Double[] Wvalues = null;
    Double[] Bvalues = null;
    File file = null;
    BufferedReader br = null;
    String st = null;

    file = new File(path + weights0file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[250];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 50; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias0file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[50];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases0 = new Double[50];
    index = 0;
    for (int k = 0; k < 50; k++) {
      biases0[k] = Bvalues[index];
      index++;
    }
    weights0 = new Double[5][50];
    index = 0;
    for (int I = 0; I < 5; I++)
      for (int J = 0; J < 50; J++)
          {
            weights0[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights1file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[2500];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 50; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias1file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[50];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases1 = new Double[50];
    index = 0;
    for (int k = 0; k < 50; k++) {
      biases1[k] = Bvalues[index];
      index++;
    }
    weights1 = new Double[50][50];
    index = 0;
    for (int I = 0; I < 50; I++)
      for (int J = 0; J < 50; J++)
          {
            weights1[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights2file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[2500];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 50; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias2file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[50];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases2 = new Double[50];
    index = 0;
    for (int k = 0; k < 50; k++) {
      biases2[k] = Bvalues[index];
      index++;
    }
    weights2 = new Double[50][50];
    index = 0;
    for (int I = 0; I < 50; I++)
      for (int J = 0; J < 50; J++)
          {
            weights2[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights3file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[2500];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 50; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias3file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[50];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases3 = new Double[50];
    index = 0;
    for (int k = 0; k < 50; k++) {
      biases3[k] = Bvalues[index];
      index++;
    }
    weights3 = new Double[50][50];
    index = 0;
    for (int I = 0; I < 50; I++)
      for (int J = 0; J < 50; J++)
          {
            weights3[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights4file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[2500];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 50; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias4file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[50];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases4 = new Double[50];
    index = 0;
    for (int k = 0; k < 50; k++) {
      biases4[k] = Bvalues[index];
      index++;
    }
    weights4 = new Double[50][50];
    index = 0;
    for (int I = 0; I < 50; I++)
      for (int J = 0; J < 50; J++)
          {
            weights4[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights5file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[2500];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 50; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias5file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[50];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases5 = new Double[50];
    index = 0;
    for (int k = 0; k < 50; k++) {
      biases5[k] = Bvalues[index];
      index++;
    }
    weights5 = new Double[50][50];
    index = 0;
    for (int I = 0; I < 50; I++)
      for (int J = 0; J < 50; J++)
          {
            weights5[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights6file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[250];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 5; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias6file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[5];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases6 = new Double[5];
    index = 0;
    for (int k = 0; k < 5; k++) {
      biases6[k] = Bvalues[index];
      index++;
    }
    weights6 = new Double[50][5];
    index = 0;
    for (int I = 0; I < 50; I++)
      for (int J = 0; J < 5; J++)
          {
            weights6[I][J] = Wvalues[index];
            index++;
          }
    br.close();

  }
}