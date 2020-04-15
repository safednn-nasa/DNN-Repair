import java.io.*;
public class InternalData {
  public Double[][] weights1;
  public Double[][] weights2;
  public Double[][] weights3;
  public Double[][] weights4;
  public Double[][] weights5;
  public Double[] biases1;
  public Double[] biases2;
  public Double[] biases3;
  public Double[] biases4;
  public Double[] biases5;

  public InternalData(String weights1file,String weights2file,String weights3file,String weights4file,String weights5file,String bias1file,String bias2file,String bias3file,String bias4file,String bias5file) throws NumberFormatException, IOException {

    String path = "./params/";
    int index = 0;
    Double[] Wvalues = null;
    Double[] Bvalues = null;
    File file = null;
    BufferedReader br = null;
    String st = null;

    file = new File(path + weights1file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[46256];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 59; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias1file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[59];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases1 = new Double[59];
    index = 0;
    for (int k = 0; k < 59; k++) {
      biases1[k] = Bvalues[index];
      index++;
    }
    weights1 = new Double[784][59];
    index = 0;
    for (int I = 0; I < 784; I++)
      for (int J = 0; J < 59; J++)
          {
            weights1[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights2file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[5546];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 94; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias2file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[94];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases2 = new Double[94];
    index = 0;
    for (int k = 0; k < 94; k++) {
      biases2[k] = Bvalues[index];
      index++;
    }
    weights2 = new Double[59][94];
    index = 0;
    for (int I = 0; I < 59; I++)
      for (int J = 0; J < 94; J++)
          {
            weights2[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights3file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[5264];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 56; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias3file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[56];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases3 = new Double[56];
    index = 0;
    for (int k = 0; k < 56; k++) {
      biases3[k] = Bvalues[index];
      index++;
    }
    weights3 = new Double[94][56];
    index = 0;
    for (int I = 0; I < 94; I++)
      for (int J = 0; J < 56; J++)
          {
            weights3[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights4file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[2520];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 45; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias4file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[45];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases4 = new Double[45];
    index = 0;
    for (int k = 0; k < 45; k++) {
      biases4[k] = Bvalues[index];
      index++;
    }
    weights4 = new Double[56][45];
    index = 0;
    for (int I = 0; I < 56; I++)
      for (int J = 0; J < 45; J++)
          {
            weights4[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights5file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[450];
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
    file = new File(path + bias5file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[10];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases5 = new Double[10];
    index = 0;
    for (int k = 0; k < 10; k++) {
      biases5[k] = Bvalues[index];
      index++;
    }
    weights5 = new Double[45][10];
    index = 0;
    for (int I = 0; I < 45; I++)
      for (int J = 0; J < 10; J++)
          {
            weights5[I][J] = Wvalues[index];
            index++;
          }
    br.close();

  }
}