package mnist0_adv;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import gov.nasa.jpf.Config;
import gov.nasa.jpf.JPF;
import gov.nasa.jpf.symbc.Debug;
public class InvokeSPF {
	public static void main(String[] args) throws IOException {
		//System.out.println("BYEBYE");
		String path="C:\\Users\\musman\\eclipse-workspace\\jpf-symbc\\src\\examples\\mnist0_adv\\";
		String []inputfiles= {"0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19"};
		String constraintfile="C:\\Users\\musman\\eclipse-workspace\\jpf-symbc\\src\\examples\\mnist0_adv\\repairFor"+Run_baseline.T_LABEL+".txt";
		File readfileconstraint = new File(constraintfile);
		String constraintstring="";
		readfileconstraint.createNewFile();
		FileWriter frwriteconstraint = new FileWriter(constraintfile);
		for(int I=0;I<128;I++) {
			if(Run_baseline.T_LABEL==0 && (I==28 || I==105 || I==33 || I==107 || I==0)) {
				frwriteconstraint.write("(declare-fun sym"+I+" () Real)\n");
				frwriteconstraint.write("(assert (> sym"+I+" -0.5))\n");
				frwriteconstraint.write("(assert (< sym"+I+" 0.5))\n");}
			if(Run_baseline.T_LABEL==1 && (I==0 || I==33 || I==41 || I==23 || I==34)) {
				frwriteconstraint.write("(declare-fun sym"+I+" () Real)\n");
				frwriteconstraint.write("(assert (> sym"+I+" -0.5))\n");
				frwriteconstraint.write("(assert (< sym"+I+" 0.5))\n");}
			if(Run_baseline.T_LABEL==2 && (I==117 || I==78 || I==98 || I==93 || I==68)) {
				frwriteconstraint.write("(declare-fun sym"+I+" () Real)\n");
				frwriteconstraint.write("(assert (> sym"+I+" -0.5))\n");
				frwriteconstraint.write("(assert (< sym"+I+" 0.5))\n");}
			if(Run_baseline.T_LABEL==3 && (I==53 || I==13 || I==95 || I==30 || I==0)) {
				frwriteconstraint.write("(declare-fun sym"+I+" () Real)\n");
				frwriteconstraint.write("(assert (> sym"+I+" -0.5))\n");
				frwriteconstraint.write("(assert (< sym"+I+" 0.5))\n");}
			if(Run_baseline.T_LABEL==4 && (I==37 || I==43 || I==101 || I==98 || I==44)) { 
				frwriteconstraint.write("(declare-fun sym"+I+" () Real)\n");
				frwriteconstraint.write("(assert (> sym"+I+" -0.5))\n");
				frwriteconstraint.write("(assert (< sym"+I+" 0.5))\n");}
			if(Run_baseline.T_LABEL==5 && (I==23 || I==10 || I==11 || I==54 || I==28)) { 
				frwriteconstraint.write("(declare-fun sym"+I+" () Real)\n");
				frwriteconstraint.write("(assert (> sym"+I+" -0.5))\n");
				frwriteconstraint.write("(assert (< sym"+I+" 0.5))\n");}
			if(Run_baseline.T_LABEL==6 && (I==64 || I==105 || I==27 || I==69 || I==38)) { 
				frwriteconstraint.write("(declare-fun sym"+I+" () Real)\n");
				frwriteconstraint.write("(assert (> sym"+I+" -0.5))\n");
				frwriteconstraint.write("(assert (< sym"+I+" 0.5))\n");}
			if(Run_baseline.T_LABEL==7 && (I==119 || I==126 || I==61 || I==114 || I==37)) { 
				frwriteconstraint.write("(declare-fun sym"+I+" () Real)\n");
				frwriteconstraint.write("(assert (> sym"+I+" -0.5))\n");
				frwriteconstraint.write("(assert (< sym"+I+" 0.5))\n");}
			if(Run_baseline.T_LABEL==8 && (I==121 || I==105 || I==23 || I==62 || I==49)) { 
				frwriteconstraint.write("(declare-fun sym"+I+" () Real)\n");
				frwriteconstraint.write("(assert (> sym"+I+" -0.5))\n");
				frwriteconstraint.write("(assert (< sym"+I+" 0.5))\n");}
			if(Run_baseline.T_LABEL==9 && (I==114 || I==93 || I==1 || I==124 || I==53)) { 
				frwriteconstraint.write("(declare-fun sym"+I+" () Real)\n");
				frwriteconstraint.write("(assert (> sym"+I+" -0.5))\n");
				frwriteconstraint.write("(assert (< sym"+I+" 0.5))\n");}
		}	
		frwriteconstraint.write("\n\n\n");
		frwriteconstraint.close();
		for (int i=0;i<inputfiles.length;i++)
		{
			Config conf = JPF.createConfig(new String[0]); // just an empty configuration
			conf.setProperty("classpath", "${jpf-symbc}/build/examples/");
			conf.setProperty("target","mnist0_adv.SymbolicDriver");
			conf.setProperty("target.args",inputfiles[i]);
			conf.setProperty("sourcepath","${jpf-symbc}/src/examples/");
			conf.setProperty("symbolic.dp","no_solver");
			conf.setProperty("symbolic.collect_constraints","true");
			conf.setProperty("symbolic.optimizechoices","false");
			JPF jpf = new JPF(conf);
			jpf.run();	
		}   
		File readfileheader = new File(constraintfile);
		FileReader frreadheader = new FileReader(readfileheader);
		BufferedReader brreadheader = new BufferedReader(frreadheader);

		//Read Constraint File
		constraintstring="";
		String line="";
		while((line = brreadheader.readLine()) != null){
			constraintstring=constraintstring+line+"\n";
		}
		//Write new constraint to file	
		File file = new File(constraintfile);
		FileWriter fr = new FileWriter(file);
		BufferedWriter br = new BufferedWriter(fr);
		br.write(constraintstring);

		for(int s=0;s<10;s++) 
		{
			for(int t=0;t<10;t++)
			{
				if(t!=Run_baseline.T_LABEL)
					br.write("(assert-soft (> y"+s+"_"+Run_baseline.T_LABEL+"  y"+s+"_"+t+"))\n");
			}
			br.write("\n");
		}
		br.write("(check-sat)\n(get-model)\n");
		br.close();
		fr.close();
	}
}
