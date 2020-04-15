import java.io.*;
public class InternalData {
  public Double[][] weights1;
  public Double[][] weights2;
  public Double[][] weights3;
  public Double[][] weights4;
  public Double[][] weights5;
  public Double[][] weights6;
  public Double[] biases1;
  public Double[] biases2;
  public Double[] biases3;
  public Double[] biases4;
  public Double[] biases5;
  public Double[] biases6;

  public InternalData(String weights1file,String weights2file,String weights3file,String weights4file,String weights5file,String weights6file,String bias1file,String bias2file,String bias3file,String bias4file,String bias5file,String bias6file) throws NumberFormatException, IOException {

    String path = "./params/";
    int index = 0;
    Double[] Wvalues = null;
    Double[] Bvalues = null;
    File file = null;
    BufferedReader br = null;
    String st = null;

    file = new File(path + weights1file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[38416];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 49; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias1file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[49];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases1 = new Double[49];
    index = 0;
    for (int k = 0; k < 49; k++) {
      biases1[k] = Bvalues[index];
      index++;
    }
    weights1 = new Double[784][49];
    index = 0;
    for (int I = 0; I < 784; I++)
      for (int J = 0; J < 49; J++)
          {
            weights1[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights2file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[2989];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 61; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias2file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[61];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases2 = new Double[61];
    index = 0;
    for (int k = 0; k < 61; k++) {
      biases2[k] = Bvalues[index];
      index++;
    }
    weights2 = new Double[49][61];
    index = 0;
    for (int I = 0; I < 49; I++)
      for (int J = 0; J < 61; J++)
          {
            weights2[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights3file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[5490];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 90; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias3file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[90];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases3 = new Double[90];
    index = 0;
    for (int k = 0; k < 90; k++) {
      biases3[k] = Bvalues[index];
      index++;
    }
    weights3 = new Double[61][90];
    index = 0;
    for (int I = 0; I < 61; I++)
      for (int J = 0; J < 90; J++)
          {
            weights3[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights4file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[1890];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 21; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias4file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[21];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases4 = new Double[21];
    index = 0;
    for (int k = 0; k < 21; k++) {
      biases4[k] = Bvalues[index];
      index++;
    }
    weights4 = new Double[90][21];
    index = 0;
    for (int I = 0; I < 90; I++)
      for (int J = 0; J < 21; J++)
          {
            weights4[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights5file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[1008];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 48; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias5file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[48];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases5 = new Double[48];
    index = 0;
    for (int k = 0; k < 48; k++) {
      biases5[k] = Bvalues[index];
      index++;
    }
    weights5 = new Double[21][48];
    index = 0;
    for (int I = 0; I < 21; I++)
      for (int J = 0; J < 48; J++)
          {
            weights5[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights6file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[480];
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
    weights6 = new Double[48][10];
    index = 0;
    for (int I = 0; I < 48; I++)
      for (int J = 0; J < 10; J++)
          {
            weights6[I][J] = Wvalues[index];
            index++;
          }
    br.close();

  }
}