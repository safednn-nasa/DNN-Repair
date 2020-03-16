package dnn;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import gov.nasa.jpf.Config;
import gov.nasa.jpf.JPF;
import gov.nasa.jpf.symbc.Debug;
public class InvokeSPF {
	public static void main(String[] args) {
		try {
			String path="C:\\Users\\usman\\eclipse-workspace\\jpf\\jpf-symbc\\src\\examples\\dnn\\";

			String constraintfilepath=path+"constraint.txt";
			File readfileconstraint = new File(constraintfilepath);
			FileReader frreadconstraint = new FileReader(readfileconstraint);
			BufferedReader brreadconstraint = new BufferedReader(frreadconstraint);

			String headerfilepath=path+"header.txt";
			File readfileheader = new File(headerfilepath);
			FileReader frreadheader = new FileReader(readfileheader);
			BufferedReader brreadheader = new BufferedReader(frreadheader);

			//Initialize Constraint File to Empty
			String constraintstring="";
			readfileconstraint.createNewFile();
			FileWriter frwriteconstraint = new FileWriter(constraintfilepath);
			frwriteconstraint.write("");
			frwriteconstraint.close();

			String line;
			String number="";
			String header="";
			while((line = brreadheader.readLine()) != null){
				number=number+line;
			}
			String[] ar=number.split(",");
			for(int i=0;i<ar.length;i++)
			{
				header=header+"(declare-fun sym_"+ar[i]+" () Real)\n";
			}
			header=header+"\n";
			for(int i=0;i<ar.length;i++)
			{
				header=header+"(assert( >= sym_"+ar[i]+" -1.0))\n";
			}
			header=header+"\n";
			for(int i=0;i<ar.length;i++)
			{
				header=header+"(assert( <= sym_"+ar[i]+" 1.0))\n";
			}
			header=header+"\n";
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//Runs SPF
			String []inputfiles= {"Fail1.csv","Fail2.csv","Fail3.csv","Fail4.csv","Fail5.csv","Fail7.csv","Fail8.csv","Fail9.csv","Fail10.csv",
								  "Pass1.csv","Pass2.csv","Pass3.csv","Pass4.csv","Pass5.csv","Pass6.csv","Pass7.csv","Pass8.csv","Pass9.csv","Pass10.csv"};
	
			for (int i=0;i<inputfiles.length;i++)
			{
				Config conf = JPF.createConfig(new String[0]); // just an empty configuration
				conf.setProperty("classpath", "${jpf-symbc}/build/examples/");
				conf.setProperty("target","dnn.SymbolicDriver");
				conf.setProperty("target.args",path+"data\\"+inputfiles[i]);
				conf.setProperty("sourcepath","${jpf-symbc}/src/examples/");
				conf.setProperty("symbolic.dp","no_solver");
				conf.setProperty("symbolic.collect_constraints","true");
				conf.setProperty("symbolic.optimizechoices","false");
				JPF jpf = new JPF(conf);
				jpf.run();	
			}
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// Adds footer for Z3
			line="";
			while((line = brreadconstraint.readLine()) != null){
				constraintstring=constraintstring+line+"\n";  
			}
			constraintstring=header+constraintstring+"\n"+"(check-sat)"+"\n"+"(get-model)";
			frwriteconstraint = new FileWriter(constraintfilepath);
			frwriteconstraint.write(constraintstring);
			frwriteconstraint.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
