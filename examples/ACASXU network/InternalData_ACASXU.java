import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class InternalData_ACASXU {

	  public Double[][] weights0 = new Double[5][50];
	  public Double[][] weights1 = new Double[50][50];
	  public Double[][] weights2 = new Double[50][50];
	  public Double[][] weights3 = new Double[50][50];
	  public Double[][] weights4 = new Double[50][50];
	  public Double[][] weights5 = new Double[50][50];
	  public Double[][] weights6 = new Double[50][5];
	 
	  public Double[] biases0 = new Double[50];
	  public Double[] biases1 = new Double[50];
	  public Double[] biases2 = new Double[50];
	  public Double[] biases3 = new Double[50];
	  public Double[] biases4 = new Double[50];
	  public Double[] biases5 = new Double[50];
	  public Double[] biases6 = new Double[5];

	  public InternalData_ACASXU(String path, String inputFile) throws NumberFormatException, IOException {

		//String path = "C:\\Users\\dgopinat\\eclipse-workspace\\mnist_example\\mnist_example\\data\\ACASXU\\";
	    int index = 0;
	    File file = null;
	    BufferedReader br = null;
	    String st = null;

	    file = new File(path + inputFile);
	    br = new BufferedReader(new FileReader(file));
	    //int currentLine = 0;
	    //int prevlayer = -1;
	    
	    while ((st = br.readLine()) != null) {
	      if (st.isEmpty()) continue;
	      String[] vals = st.split(",");
	      int layer = Integer.parseInt(vals[0]);
	      int isBias = Integer.parseInt(vals[1]);
	      int OutNum = Integer.parseInt(vals[2]);
	      int InNum = Integer.parseInt(vals[3]) -1;
	      
	     // if ((prevlayer == -1) || (prevlayer != layer))
	     // {
             /** if (prevlayer == 0)
                  weightMatrix[prevlayer] = weightMatrix_layer0
                  biasMatrix[prevlayer] = biasMatrix_layer0
              if (prevlayer == 1):
                  weightMatrix[prevlayer] = weightMatrix_layer1
                  biasMatrix[prevlayer] = biasMatrix_layer1
              if (prevlayer == 2):
                  weightMatrix[prevlayer] = weightMatrix_layer2
                  biasMatrix[prevlayer] = biasMatrix_layer2
              if (prevlayer == 3):
                  weightMatrix[prevlayer] = weightMatrix_layer3
                  biasMatrix[prevlayer] = biasMatrix_layer3
              if (prevlayer == 4):
                  weightMatrix[prevlayer] = weightMatrix_layer4
                  biasMatrix[prevlayer] = biasMatrix_layer4
              if (prevlayer == 5):
                  weightMatrix[prevlayer] = weightMatrix_layer5
                  biasMatrix[prevlayer] = biasMatrix_layer5**/
           //   prevlayer = layer;
	      //}
	      
	      if (isBias == 0)
	      {
	    	    if (layer == 0)
                   weights0[InNum][OutNum] = Double.parseDouble(vals[4]);
                if (layer == 1)
                   weights1[InNum][OutNum] = Double.parseDouble(vals[4]);
                if (layer == 2)
                	weights2[InNum][OutNum] = Double.parseDouble(vals[4]);
                if (layer == 3)
                	weights3[InNum][OutNum] = Double.parseDouble(vals[4]);
                if (layer == 4)
                	weights4[InNum][OutNum] = Double.parseDouble(vals[4]);
                if (layer == 5)
                	weights5[InNum][OutNum] = Double.parseDouble(vals[4]);
                if (layer == 6)
                    weights6[InNum][OutNum] = Double.parseDouble(vals[4]);  
	    		  
	      }
	      else 
	      {
              if (layer == 0)
                biases0[OutNum] = Double.parseDouble(vals[4]);
              if (layer == 1)
            	biases1[OutNum] = Double.parseDouble(vals[4]);
              if (layer == 2)
            	biases2[OutNum] = Double.parseDouble(vals[4]);
              if (layer == 3)
            	biases3[OutNum] = Double.parseDouble(vals[4]);
              if (layer == 4)
            	biases4[OutNum] = Double.parseDouble(vals[4]);
              if (layer == 5)
            	biases5[OutNum] = Double.parseDouble(vals[4]);
              if (layer == 6)
            	biases6[OutNum] = Double.parseDouble(vals[4]);
	      }
	      
	    }
	    br.close();
	    

	  }
	
}
