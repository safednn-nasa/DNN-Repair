import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Experiment Execution - Testing
 */
public class MNISTExperiments {
	
	/**
	 * Enumeration for supported models.
	 */
	enum SUBJECT {

		// LOW_QUALITY(workspace + "data\\low-quality\\"),
		// LOW_QUALITY("/Users/yannic/repositories/DNN-Repair/experiments/pattern-based-repair/"),
		// POISONED(workspace + "data\\poisoned\\"),
		// HIGH_QUALITY(workspace + "data\\high-quality\\");

		LOW_QUALITY_PATTERN("/Users/yannic/repositories/DNN-Repair/experiments/pattern-based-repair/low-quality", 6),
		LOW_QUALITY_LAST_LAYER("/Users/yannic/repositories/DNN-Repair/experiments/last-layer-usman", 8);

		private String path;
		private int repairedLayerId;

		SUBJECT(String path, int repairedLayerId) {
			this.path = path;
			this.repairedLayerId = repairedLayerId;
		}

		public String getPath() {
			return path;
		}

		public int getRepairedLayerId() {
			return repairedLayerId;
		}
	}
	
	/*
	 * *****************************************************************************
	 * Utilities
	 * *****************************************************************************
	 */

	private static double round(double value, int places) {
		if (places < 0)
			throw new IllegalArgumentException();
		BigDecimal bd = BigDecimal.valueOf(value);
		bd = bd.setScale(places, RoundingMode.HALF_UP);
		return bd.doubleValue();
	}
	
	
	/*
	 * *****************************************************************************
	 * Experiment Definitions
	 * *****************************************************************************
	 */
	
	public static void runExperiment() throws NumberFormatException, IOException {

		/* ************************************************************** */
		/* Parameters for the experiment. */

		SUBJECT subject = SUBJECT.LOW_QUALITY_LAST_LAYER;

		DNNtCombined.COMBINATION_METHOD combMethod = DNNtCombined.COMBINATION_METHOD.ALL;
		
		int repairedLayerId = subject.getRepairedLayerId(); // {0 | 2 | 6 | 8}

		String labelFile = subject.getPath() + "/mnist_test_labels.txt";
		String inputFile = subject.getPath() + "/mnist_test.txt";
//		String labelFile = subject.getPath() + "/mnist_train_labels.txt";
//		String inputFile = subject.getPath() + "/mnist_train.txt";

		int stopAfter = 60000; // 60000;

		// training precision
//		double[] trainPrecision = new double[] { 0.970734126984127, 0.9848417298261257, 0.6285016977928692,
//				0.950096587250483, 0.6775396085740913, 0.914295509084676, 0.9900068917987594, 0.9570011025358324,
//				0.9746994437466355, 0.9322475570032573 }; // low_quality pattern_based layer_6
		double[] trainPrecision = new double[] { 0.9346001583531275, 0.7040612865988036, 0.7582460225067909,
				0.3880053650124545, 0.7409247229652274, 0.32622190080154284, 0.8278192313107138, 0.9191318566968931,
				0.8989931276969794, 0.4081175647305808 }; // low_quality last_layer

		// Experts whose F1 score on train set is more than original model.
		boolean useF1Selection = false;
		int[] f1SelectedExperts = new int[] { 6, 8, 9 };
		
		// Optimization does not calculate Average and Full iterations.
		boolean optimized = false;

		/* ************************************************************** */

		System.out.println("PATH:" + subject.getPath());

		InternalData data = new InternalData(subject.getPath(), "weights0.txt", "weights2.txt", "weights6.txt",
				"weights8.txt", "biases0.txt", "biases2.txt", "biases6.txt", "biases8.txt");
		Object repaired_weight_deltas = Z3SolutionParsing.loadRepairedWeights(subject.getPath(), repairedLayerId, DNNtCombined.NUMBER_OF_EXPERTS);
		DNNtCombined model = new DNNtCombined(data, repaired_weight_deltas);

		/* Initialize analytics */
		Map<Object, Integer> passCounter = new HashMap<>();
		Map<Object, Integer> failCounter = new HashMap<>();
		Map<Object, Integer> targetedPassCounter = new HashMap<>();
		Map<Object, Integer> targetedFailCounter = new HashMap<>();
		Map<Object, Integer> TPCounter = new HashMap<>();
		Map<Object, Integer> TNCounter = new HashMap<>();
		Map<Object, Integer> FPCounter = new HashMap<>();
		Map<Object, Integer> FNCounter = new HashMap<>();
		for (DNNtCombined.COMBINATION_METHOD x : DNNtCombined.COMBINATION_METHOD.values()) {
			passCounter.put(x, 0);
			failCounter.put(x, 0);
		}
		for (int x = 0; x < DNNtCombined.NUMBER_OF_EXPERTS + 2; x++) {
			passCounter.put(x, 0);
			failCounter.put(x, 0);
			targetedPassCounter.put(x, 0);
			targetedFailCounter.put(x, 0);
			TPCounter.put(x, 0);
			TNCounter.put(x, 0);
			FPCounter.put(x, 0);
			FNCounter.put(x, 0);
		}

		/* Read correct labels. */
		File file = new File(labelFile);
		BufferedReader br = new BufferedReader(new FileReader(file));
		String st;
		Integer[] labels = new Integer[60000];
		int index = 0;
		while ((st = br.readLine()) != null) {
			labels[index] = Integer.valueOf(st);
			index++;
			if (index == stopAfter) {
				break;
			}
		}
		br.close();

		/* Prepare the experts. */
		int[] expertIDs;
		if (useF1Selection) {
			expertIDs = f1SelectedExperts;
		} else {
			expertIDs = new int[DNNtCombined.NUMBER_OF_EXPERTS];
			for (int i = 0; i < DNNtCombined.NUMBER_OF_EXPERTS; i++) {
				expertIDs[i] = i;
			}
		}

		/* Read input files and execute model. */
		file = new File(inputFile);
		br = new BufferedReader(new FileReader(file));
		int count = 0;

		while ((st = br.readLine()) != null) {
			// System.out.println("INPUT:" + st);

			String[] values = st.split(",");
			double[][][] input = new double[28][28][1];
			index = 0;
			while (index < values.length) {
				for (int i = 0; i < 28; i++)
					for (int j = 0; j < 28; j++)
						for (int k = 0; k < 1; k++) {
							Double val = Double.valueOf(values[index]);
							index++;
							// input[i][j][k] = (double)(val/255.0);
							input[i][j][k] = (double) (val);
						}
			}

			Map<Integer, double[]> result = model.run(input, repairedLayerId, expertIDs, optimized);

			int correctLabel = labels[count];

			// Extract original decision.
			int origLabel = DNNtCombined.selectLabelWithMaxConfidence(result.get(-1)); /* ORIG */

			// Determine final decisions by experts.
			Map<DNNtCombined.COMBINATION_METHOD, Integer> results = DNNtCombined.combineExperts(combMethod, result, origLabel, trainPrecision,
					expertIDs, optimized);

			// Print results and collect analytics.
			System.out.print(count + "; IDEAL: " + correctLabel + "; ");
			for (Entry<DNNtCombined.COMBINATION_METHOD, Integer> combinedResult : results.entrySet()) {
				DNNtCombined.COMBINATION_METHOD currentCombinationMethod = combinedResult.getKey();
				int label = combinedResult.getValue();

				boolean passed = (label == correctLabel);
				if (passed) {
					passCounter.put(currentCombinationMethod, passCounter.get(currentCombinationMethod) + 1);
				} else {
					failCounter.put(currentCombinationMethod, failCounter.get(currentCombinationMethod) + 1);
				}

				System.out.print(currentCombinationMethod + ": " + (passed ? "PASS" : "FAIL") + " " + label + "; ");
			}

			// Collect results for experts. Accuracy is only interesting for targeted
			// repair. Precision is wanted for all experts.
			for (int expertId = 0; expertId < DNNtCombined.NUMBER_OF_EXPERTS; expertId++) {
				int label = DNNtCombined.selectLabelWithMaxConfidence(result.get(expertId));
				boolean passed = (label == correctLabel);
				if (passed) {
					passCounter.put(expertId, passCounter.get(expertId) + 1);
					if (correctLabel == expertId) {
						TPCounter.put(expertId, TPCounter.get(expertId) + 1);

						targetedPassCounter.put(expertId, targetedPassCounter.get(expertId) + 1); // only for local
																									// expert
						System.out.print("ExpertL" + expertId + ": " + "PASS" + " " + label + "; ");
					} else {
						TNCounter.put(expertId, TNCounter.get(expertId) + 1);
					}
				} else {
					failCounter.put(expertId, failCounter.get(expertId) + 1);
					if (correctLabel == expertId) {
						FNCounter.put(expertId, FNCounter.get(expertId) + 1);

						targetedFailCounter.put(expertId, targetedFailCounter.get(expertId) + 1); // only for local
																									// expert
						System.out.print("ExpertL" + expertId + ": " + "FAIL" + " " + label + "; ");
					} else if (label == expertId) {
						FPCounter.put(expertId, FPCounter.get(expertId) + 1);
					} else {
						TNCounter.put(expertId, TNCounter.get(expertId) + 1);
					}
				}
			}

			System.out.println();
			count++;

			if (count == stopAfter) {
				break;
			}

		}

		br.close();

		// Calculate and print accuracy.
		System.out.println();
		System.out.println("COMBINATION;ACCURACY;PASS;FAIL;TAR-ACC;TAR-PASS;TAR-FAIL;TP;TN;FP;FN;PREC;RECALL;F1");
		if (combMethod.equals(DNNtCombined.COMBINATION_METHOD.ALL)) {
			for (DNNtCombined.COMBINATION_METHOD combinationMethod : DNNtCombined.COMBINATION_METHOD.values()) {
				if (combinationMethod.equals(DNNtCombined.COMBINATION_METHOD.ALL)) {
					continue;
				}
				int pass = passCounter.get(combinationMethod);
				int fail = failCounter.get(combinationMethod);
				double accuracy = round((((double) pass) / (pass + fail)) * 100.0, 2);

				System.out.println(combinationMethod + ";" + accuracy + ";" + pass + ";" + fail + ";;;;;;;;;;");
			}
		} else {
			int pass = passCounter.get(combMethod);
			int fail = failCounter.get(combMethod);
			double accuracy = round((((double) pass) / (pass + fail)) * 100.0, 2);

			System.out.println(combMethod + ";" + accuracy + ";" + pass + ";" + fail);
		}
		double[] prec = new double[DNNtCombined.NUMBER_OF_EXPERTS];
		for (int expertId = 0; expertId < DNNtCombined.NUMBER_OF_EXPERTS; expertId++) {
			int pass = passCounter.get(expertId);
			int fail = failCounter.get(expertId);
			double accuracy = round((((double) pass) / (pass + fail)) * 100.0, 2);

			int targetedPass = targetedPassCounter.get(expertId);
			int targetedFail = targetedFailCounter.get(expertId);
			double targetedAccuracy = round((((double) targetedPass) / (targetedPass + targetedFail)) * 100.0, 2);

			int TP = TPCounter.get(expertId);
			int TN = TNCounter.get(expertId);
			int FP = FPCounter.get(expertId);
			int FN = FNCounter.get(expertId);
			double precision = ((double) TP) / (TP + FP);
			prec[expertId] = precision;

			double recall = ((double) TP) / (TP + FN);
			double f1 = 2 * precision * recall / (precision + recall);

			System.out.println("L" + expertId + ";" + accuracy + ";" + pass + ";" + fail + ";" + targetedAccuracy + ";"
					+ targetedPass + ";" + targetedFail + ";" + TP + ";" + TN + ";" + FP + ";" + FN + ";"
					+ round(precision * 100.0, 2) + ";" + round(recall * 100.0, 2) + ";" + round(f1 * 100.0, 2));
		}
		System.out.println();
		System.out.println("prec=" + Arrays.toString(prec));

	}

	public static void runCombinationOverheadExperiment() throws NumberFormatException, IOException {

		/* ************************************************************** */
		/* Parameters for the experiment. */

		SUBJECT subject = SUBJECT.LOW_QUALITY_LAST_LAYER;
		int repairedLayerId = subject.getRepairedLayerId();

		String labelFile = subject.getPath() + "/mnist_test_labels.txt";
		String inputFile = subject.getPath() + "/mnist_test.txt";
//		String labelFile = subject.getPath() + "/mnist_train_labels.txt";
//		String inputFile = subject.getPath() + "/mnist_train.txt";

		int stopAfter = 60000; // 60000;

		// training precision
		double[] trainPrecision = new double[] { 0.970734126984127, 0.9848417298261257, 0.6285016977928692,
				0.950096587250483, 0.6775396085740913, 0.914295509084676, 0.9900068917987594, 0.9570011025358324,
				0.9746994437466355, 0.9322475570032573 }; // low_quality pattern_based layer_6

		// Experts whose F1 score on train set is more than original model.
		boolean useF1Selection = false;
		int[] f1SelectedExperts = new int[] { 6, 8, 9 };
		
		// Optimization does not calculate Average and Full iterations.
		boolean optimized = true;

		/* ************************************************************** */

		System.out.println("PATH:" + subject.getPath());

		InternalData data = new InternalData(subject.getPath(), "weights0.txt", "weights2.txt", "weights6.txt",
				"weights8.txt", "biases0.txt", "biases2.txt", "biases6.txt", "biases8.txt");
		Object repaired_weight_deltas = Z3SolutionParsing.loadRepairedWeights(subject.getPath(), repairedLayerId, DNNtCombined.NUMBER_OF_EXPERTS);
		DNNtCombined model = new DNNtCombined(data, repaired_weight_deltas);
		DNNtOriginal origModel = new DNNtOriginal(data);

		/* Initialize analytics */
		long accumulatedTimeOriginal = 0;
		long accumulatedTimeCombinedNetwork = 0;
		long accumulatedTimeNAIVE = 0;
		long accumulatedTimeNAIVETotal = 0;
		long accumulatedTimePREC = 0;
		long accumulatedTimePRECTotal = 0;
		long accumulatedTimeVOTES = 0;
		long accumulatedTimeVOTESTotal = 0;
		long accumulatedTimeCONF = 0;
		long accumulatedTimeCONFTotal = 0;
		long accumulatedTimePVC = 0;
		long accumulatedTimePVCTotal = 0;

		/* Read correct labels. */
		File file = new File(labelFile);
		BufferedReader br = new BufferedReader(new FileReader(file));
		String st;
		Integer[] labels = new Integer[60000];
		int index = 0;
		while ((st = br.readLine()) != null) {
			labels[index] = Integer.valueOf(st);
			index++;
			if (index == stopAfter) {
				break;
			}
		}
		br.close();

		/* Prepare the experts. */
		int[] expertIDs;
		if (useF1Selection) {
			expertIDs = f1SelectedExperts;
		} else {
			expertIDs = new int[DNNtCombined.NUMBER_OF_EXPERTS];
			for (int i = 0; i < DNNtCombined.NUMBER_OF_EXPERTS; i++) {
				expertIDs[i] = i;
			}
		}

		/* Read input files and execute model. */
		file = new File(inputFile);
		br = new BufferedReader(new FileReader(file));
		int count = 0;

		while ((st = br.readLine()) != null) {
			// System.out.println("INPUT:" + st);

			String[] values = st.split(",");
			double[][][] input = new double[28][28][1];
			index = 0;
			while (index < values.length) {
				for (int i = 0; i < 28; i++)
					for (int j = 0; j < 28; j++)
						for (int k = 0; k < 1; k++) {
							Double val = Double.valueOf(values[index]);
							index++;
							// input[i][j][k] = (double)(val/255.0);
							input[i][j][k] = (double) (val);
						}
			}

			System.out.print(count);

			// Run original model.
			long startTimeOriginal = System.currentTimeMillis();
			origModel.run(input);
			long timeOriginal = System.currentTimeMillis() - startTimeOriginal;
			accumulatedTimeOriginal += timeOriginal;
			System.out.print("; ORIG=" + timeOriginal);

			// Run combination.
			long startTimeCombinationNetwork = System.currentTimeMillis();
			Map<Integer, double[]> result = model.run(input, repairedLayerId, expertIDs, optimized);
			long timeCombinationNetwork = System.currentTimeMillis() - startTimeCombinationNetwork;
			accumulatedTimeCombinedNetwork += timeCombinationNetwork;

			// Combine NAIVE.
			long startTimeCombinationNAIVE = System.currentTimeMillis();
			int origLabelNAIVE = DNNtCombined.selectLabelWithMaxConfidence(result.get(-1));
			List<Integer> expertClaimsNAIVE = DNNtCombined.collectExpertClaims(expertIDs, result);
			DNNtCombined.combineExpertsByNaive(expertClaimsNAIVE, origLabelNAIVE);
			long timeNAIVE = System.currentTimeMillis() - startTimeCombinationNAIVE;
			accumulatedTimeNAIVE += timeNAIVE;
			accumulatedTimeNAIVETotal += (timeNAIVE + timeCombinationNetwork);
			System.out.print("; NAIVE=" + timeNAIVE + "; NAIVETotal=" + (timeNAIVE + timeCombinationNetwork));

			// Combine PREC.
			long startTimeCombinationPREC = System.currentTimeMillis();
			int origLabelPREC = DNNtCombined.selectLabelWithMaxConfidence(result.get(-1));
			List<Integer> expertClaimsPREC = DNNtCombined.collectExpertClaims(expertIDs, result);
			DNNtCombined.combineExpertsByPrecision(expertClaimsPREC, origLabelPREC, trainPrecision);
			long timePREC = System.currentTimeMillis() - startTimeCombinationPREC;
			accumulatedTimePREC += timePREC;
			accumulatedTimePRECTotal += (timePREC + timeCombinationNetwork);
			System.out.print("; PREC=" + timePREC + "; PRECTotal=" + (timePREC + timeCombinationNetwork));

			// Combine VOTES.
			long startTimeCombinationVOTES = System.currentTimeMillis();
			int origLabelVOTES = DNNtCombined.selectLabelWithMaxConfidence(result.get(-1));
			List<Integer> expertClaimsVOTES = DNNtCombined.collectExpertClaims(expertIDs, result);
			DNNtCombined.combineExpertsByVotes(result, expertClaimsVOTES, origLabelVOTES, expertIDs);
			long timeVOTES = System.currentTimeMillis() - startTimeCombinationVOTES;
			accumulatedTimeVOTES += timeVOTES;
			accumulatedTimeVOTESTotal += (timeVOTES + timeCombinationNetwork);
			System.out.print("; VOTES=" + timeVOTES + "; VOTESTotal=" + (timeVOTES + timeCombinationNetwork));

			// Combine CONF.
			long startTimeCombinationCONF = System.currentTimeMillis();
			int origLabelCONF = DNNtCombined.selectLabelWithMaxConfidence(result.get(-1));
			List<Integer> expertClaimsCONF = DNNtCombined.collectExpertClaims(expertIDs, result);
			DNNtCombined.combineExpertsByConfidence(result, expertClaimsCONF, origLabelCONF);
			long timeCONF = System.currentTimeMillis() - startTimeCombinationCONF;
			accumulatedTimeCONF += timeCONF;
			accumulatedTimeCONFTotal += (timeCONF + timeCombinationNetwork);
			System.out.print("; CONF=" + timeCONF + "; CONFTotal=" + (timeCONF + timeCombinationNetwork));

			// Combine PVC.
			long startTimeCombinationPVC = System.currentTimeMillis();
			int origLabelPVC = DNNtCombined.selectLabelWithMaxConfidence(result.get(-1));
			List<Integer> expertClaimsPVC = DNNtCombined.collectExpertClaims(expertIDs, result);
			DNNtCombined.combineExpertsByPVC(result, expertClaimsPVC, origLabelPVC, trainPrecision, expertIDs);
			long timePVC = System.currentTimeMillis() - startTimeCombinationPVC;
			accumulatedTimePVC += timePVC;
			accumulatedTimePVCTotal += (timePVC + timeCombinationNetwork);
			System.out.print("; PVC=" + timePVC + "; PVCTotal=" + (timePVC + timeCombinationNetwork));

			System.out.println();
			count++;

			if (count == stopAfter) {
				break;
			}

		}

		br.close();

		// Calculate and print average times.
		System.out.println();
		System.out.println("Average execution times after " + count + " inputs:");
		System.out.println();
		System.out.println("SUBJECT;AVG_TIME(ms)");
		System.out.println("ORIG;" + ((double) accumulatedTimeOriginal / count));
		System.out.println(";");
		System.out.println("COMBINED_NETWORK;" + ((double) accumulatedTimeCombinedNetwork / count));
		System.out.println(";");
		System.out.println("NAIVE;" + ((double) accumulatedTimeNAIVE / count));
		System.out.println("PREC;" + ((double) accumulatedTimePREC / count));
		System.out.println("VOTES;" + ((double) accumulatedTimeVOTES / count));
		System.out.println("CONF;" + ((double) accumulatedTimeCONF / count));
		System.out.println("PVC;" + ((double) accumulatedTimePVC / count));
		System.out.println(";");
		System.out.println("NAIVETotal;" + ((double) accumulatedTimeNAIVETotal / count));
		System.out.println("PRECTotal;" + ((double) accumulatedTimePRECTotal / count));
		System.out.println("VOTESTotal;" + ((double) accumulatedTimeVOTESTotal / count));
		System.out.println("CONFTotal;" + ((double) accumulatedTimeCONFTotal / count));
		System.out.println("PVCTotal;" + ((double) accumulatedTimePVCTotal / count));
		System.out.println();

	}

	public static void main(String[] args) {
		try {
			runExperiment();
//			runCombinationOverheadExperiment();
		} catch (NumberFormatException | IOException e) {
			e.printStackTrace();
		}
	}


}
