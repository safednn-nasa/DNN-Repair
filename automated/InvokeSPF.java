package dnn;
import gov.nasa.jpf.Config;
import gov.nasa.jpf.JPF;

public class InvokeSPF {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		try {
			String path="C:\\Users\\usman\\eclipse-workspace\\jpf\\jpf-symbc\\src\\examples\\dnn\\data\\";
			String []inputfiles= {"Fail1.csv","Fail2.csv","Fail3.csv","Fail4.csv","Fail5.csv","Fail7.csv","Fail8.csv","Fail9.csv","Fail10.csv"};
			for (int i=0;i<9;i++)
			{
			Config conf = JPF.createConfig(new String[0]); // just an empty configuration
			conf.setProperty("classpath", "${jpf-symbc}/build/examples/");
			conf.setProperty("target","dnn.SymbolicDriver");
			conf.setProperty("target.args",path+inputfiles[i]);

			conf.setProperty("sourcepath","${jpf-symbc}/src/examples/");
			conf.setProperty("symbolic.dp","no_solver");
			conf.setProperty("symbolic.collect_constraints","true");
			conf.setProperty("symbolic.optimizechoices","false");
			JPF jpf = new JPF(conf);
			jpf.run();	
			}
		
		} catch (Exception e) {
			e.printStackTrace();
		}


	}

}
