package dnn_poisoned;
import gov.nasa.jpf.Config;
import gov.nasa.jpf.JPF;

public class InvokeSPF {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Config conf;
		JPF jpf = new JPF("/Users/corinapasareanu/workspace-github/jpf-symbc/src/examples/dnn_poisoned/SPF-DNN.jpf");
        
        try {
                jpf.run();
        } catch (Exception e) {
                e.printStackTrace();
        }

		
	}

}
