
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * DNNt program that takes repaired weights as input (currently as z3 output).
 */
public class DNNtGeneralCombination {

	/*
	 * *****************************************************************************
	 * Repaired Network Implementation
	 * *****************************************************************************
	 */

	private final static int NUMBER_OF_EXPERTS = 10;

	private InternalData internal;

	private Object weight_delta;

	public DNNtGeneralCombination(InternalData internal, Object repaired_weight_deltas) throws IOException {
		this.internal = internal;
		this.weight_delta = repaired_weight_deltas;
	}

	Map<Integer, double[][][]> layer_0(double[][][] input, int repairedLayerId) {
		final int CURRENT_LAYER = 0;
		Map<Integer, double[][][]> layer0_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[][][] layer0_orig = new double[26][26][2];
		for (int i = 0; i < 26; i++)
			for (int j = 0; j < 26; j++)
				for (int k = 0; k < 2; k++) {
					layer0_orig[i][j][k] = internal.biases0[k];
					for (int I = 0; I < 3; I++)
						for (int J = 0; J < 3; J++)
							for (int K = 0; K < 1; K++)
								layer0_orig[i][j][k] += internal.weights0[I][J][K][k] * input[i + I][j + J][K];
				}
		layer0_perExpert.put(-1, layer0_orig);

		if (repairedLayerId == CURRENT_LAYER) {

			/*
			 * If the repair starts in this layer, then the layer will calculate the output
			 * with the adjusted weights for each expert.
			 */

			double[][][][] delta_layer0_perExpert = (double[][][][]) weight_delta;

			for (int expertId = 0; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {

				double[][][] layer0 = new double[26][26][2];
				for (int i = 0; i < 26; i++)
					for (int j = 0; j < 26; j++)
						for (int k = 0; k < 2; k++) {
							layer0[i][j][k] = internal.biases0[k];
							for (int I = 0; I < 3; I++)
								for (int J = 0; J < 3; J++)
									for (int K = 0; K < 1; K++)
										layer0[i][j][k] += (internal.weights0[I][J][K][k]
												+ delta_layer0_perExpert[expertId][i][j][k]) * input[i + I][j + J][K];
						}

				layer0_perExpert.put(expertId, layer0);
			}
		}

		return layer0_perExpert;
	}

	Map<Integer, double[][][]> layer_1(Map<Integer, double[][][]> layer0_perExpert, int repairedLayerId) {
		final int CURRENT_LAYER = 1;
		Map<Integer, double[][][]> layer1_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[][][] layer0_orig = layer0_perExpert.get(-1);
		double[][][] layer1_orig = new double[26][26][2];
		for (int i = 0; i < 26; i++)
			for (int j = 0; j < 26; j++)
				for (int k = 0; k < 2; k++)
					if (layer0_orig[i][j][k] > 0) {
						layer1_orig[i][j][k] = layer0_orig[i][j][k];

					} else {
						layer1_orig[i][j][k] = 0;

					}
		layer1_perExpert.put(-1, layer1_orig);

		if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert. Note: the repair cannot happen in this layer because
			 * no weights are involved.
			 */

			for (int expertId = 0; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
				double[][][] layer0 = layer0_perExpert.get(expertId);

				double[][][] layer1 = new double[26][26][2];
				for (int i = 0; i < 26; i++)
					for (int j = 0; j < 26; j++)
						for (int k = 0; k < 2; k++)
							if (layer0[i][j][k] > 0) {
								layer1[i][j][k] = layer0[i][j][k];

							} else {
								layer1[i][j][k] = 0;

							}

				layer1_perExpert.put(expertId, layer0);
			}
		}

		return layer1_perExpert;
	}

	Map<Integer, double[][][]> layer_2(Map<Integer, double[][][]> layer1_perExpert, int repairedLayerId) {
		final int CURRENT_LAYER = 2;
		Map<Integer, double[][][]> layer2_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[][][] layer1_orig = layer1_perExpert.get(-1);
		double[][][] layer2_orig = new double[24][24][4];
		for (int i = 0; i < 24; i++)
			for (int j = 0; j < 24; j++)
				for (int k = 0; k < 4; k++) {
					layer2_orig[i][j][k] = internal.biases2[k];
					for (int I = 0; I < 3; I++)
						for (int J = 0; J < 3; J++)
							for (int K = 0; K < 2; K++)
								layer2_orig[i][j][k] += internal.weights2[I][J][K][k] * layer1_orig[i + I][j + J][K];
				}
		layer2_perExpert.put(-1, layer2_orig);

		if (repairedLayerId == CURRENT_LAYER) {

			/*
			 * If the repair starts in this layer, then the layer will calculate the output
			 * with the adjusted weights for each expert.
			 */

			double[][][][][] delta_layer2_perExpert = (double[][][][][]) weight_delta;

			for (int expertId = 0; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
				double[][][] layer2 = new double[24][24][4];
				for (int i = 0; i < 24; i++)
					for (int j = 0; j < 24; j++)
						for (int k = 0; k < 4; k++) {
							layer2[i][j][k] = internal.biases2[k];
							for (int I = 0; I < 3; I++)
								for (int J = 0; J < 3; J++)
									for (int K = 0; K < 2; K++)
										layer2[i][j][k] += (internal.weights2[I][J][K][k]
												+ delta_layer2_perExpert[expertId][I][J][K][k])
												* layer1_orig[i + I][j + J][K];
						}

				layer2_perExpert.put(expertId, layer2);
			}

		} else if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert.
			 */

			for (int expertId = 0; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
				double[][][] layer1 = layer1_perExpert.get(expertId);

				double[][][] layer2 = new double[24][24][4];
				for (int i = 0; i < 24; i++)
					for (int j = 0; j < 24; j++)
						for (int k = 0; k < 4; k++) {
							layer2[i][j][k] = internal.biases2[k];
							for (int I = 0; I < 3; I++)
								for (int J = 0; J < 3; J++)
									for (int K = 0; K < 2; K++)
										layer2[i][j][k] += internal.weights2[I][J][K][k] * layer1[i + I][j + J][K];
						}

				layer2_perExpert.put(expertId, layer2);
			}
		}

		return layer2_perExpert;
	}

	Map<Integer, double[][][]> layer_3(Map<Integer, double[][][]> layer2_perExpert, int repairedLayerId) {
		final int CURRENT_LAYER = 3;
		Map<Integer, double[][][]> layer3_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[][][] layer2_orig = layer2_perExpert.get(-1);
		double[][][] layer3_orig = new double[24][24][4];
		for (int i = 0; i < 24; i++)
			for (int j = 0; j < 24; j++)
				for (int k = 0; k < 4; k++)
					if (layer2_orig[i][j][k] > 0) {
						layer3_orig[i][j][k] = layer2_orig[i][j][k];

					} else {
						layer3_orig[i][j][k] = 0;

					}
		layer3_perExpert.put(-1, layer3_orig);

		if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert. Note: the repair cannot happen in this layer because
			 * no weights are involved.
			 */

			for (int expertId = 0; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
				double[][][] layer2 = layer2_perExpert.get(expertId);

				double[][][] layer3 = new double[24][24][4];
				for (int i = 0; i < 24; i++)
					for (int j = 0; j < 24; j++)
						for (int k = 0; k < 4; k++)
							if (layer2[i][j][k] > 0) {
								layer3[i][j][k] = layer2[i][j][k];

							} else {
								layer3[i][j][k] = 0;

							}

				layer3_perExpert.put(expertId, layer3);
			}
		}

		return layer3_perExpert;
	}

	Map<Integer, double[][][]> layer_4(Map<Integer, double[][][]> layer3_perExpert, int repairedLayerId) {
		final int CURRENT_LAYER = 4;
		Map<Integer, double[][][]> layer4_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[][][] layer3_orig = layer3_perExpert.get(-1);
		double[][][] layer4_orig = new double[12][12][4];
		for (int i = 0; i < 12; i++)
			for (int j = 0; j < 12; j++)
				for (int k = 0; k < 4; k++) {
					layer4_orig[i][j][k] = 0;
					for (int I = i * 2; I < (i + 1) * 2; I++)
						for (int J = j * 2; J < (j + 1) * 2; J++)
							if (layer3_orig[I][J][k] > layer4_orig[i][j][k])
								layer4_orig[i][j][k] = layer3_orig[I][J][k];
				}
		layer4_perExpert.put(-1, layer4_orig);

		if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert. Note: the repair cannot happen in this layer because
			 * no weights are involved.
			 */

			for (int expertId = 0; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
				double[][][] layer3 = layer3_perExpert.get(expertId);

				double[][][] layer4 = new double[12][12][4];
				for (int i = 0; i < 12; i++)
					for (int j = 0; j < 12; j++)
						for (int k = 0; k < 4; k++) {
							layer4[i][j][k] = 0;
							for (int I = i * 2; I < (i + 1) * 2; I++)
								for (int J = j * 2; J < (j + 1) * 2; J++)
									if (layer3[I][J][k] > layer4[i][j][k])
										layer4[i][j][k] = layer3[I][J][k];
						}

				layer4_perExpert.put(expertId, layer3);
			}
		}

		return layer4_perExpert;
	}

	Map<Integer, double[]> layer_5(Map<Integer, double[][][]> layer4_perExpert, int repairedLayerId) {
		final int CURRENT_LAYER = 5;
		Map<Integer, double[]> layer5_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[][][] layer4_orig = layer4_perExpert.get(-1);
		double[] layer5_orig = new double[576];
		for (int i = 0; i < 576; i++) {
			int d0 = i / 48;
			int d1 = (i % 48) / 4;
			int d2 = i - d0 * 48 - d1 * 4;
			layer5_orig[i] = layer4_orig[d0][d1][d2];
		}
		layer5_perExpert.put(-1, layer5_orig);

		if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert. Note: the repair cannot happen in this layer because
			 * no weights are involved.
			 */

			for (int expertId = 0; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
				double[][][] layer4 = layer4_perExpert.get(expertId);

				double[] layer5 = new double[576];
				for (int i = 0; i < 576; i++) {
					int d0 = i / 48;
					int d1 = (i % 48) / 4;
					int d2 = i - d0 * 48 - d1 * 4;
					layer5[i] = layer4[d0][d1][d2];
				}

				layer5_perExpert.put(expertId, layer5);
			}
		}

		return layer5_perExpert;
	}

	Map<Integer, double[]> layer_6(Map<Integer, double[]> layer5_perExpert, int repairedLayerId) {
		final int CURRENT_LAYER = 6;
		Map<Integer, double[]> layer6_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[] layer5_orig = layer5_perExpert.get(-1);
		double[] layer6_orig = new double[128];
		for (int i = 0; i < 128; i++) {
			layer6_orig[i] = internal.biases6[i];
			for (int I = 0; I < 576; I++)
				layer6_orig[i] += internal.weights6[I][i] * layer5_orig[I];
		}
		layer6_perExpert.put(-1, layer6_orig);

		if (repairedLayerId == CURRENT_LAYER) {

			/*
			 * If the repair starts in this layer, then the layer will calculate the output
			 * with the adjusted weights for each expert.
			 */

			double[][][] delta_layer6_perExpert = (double[][][]) weight_delta;

			for (int expertId = 0; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
				double[] layer6 = new double[128];
				for (int i = 0; i < 128; i++) {
					layer6[i] = internal.biases6[i];
					for (int I = 0; I < 576; I++)
						layer6[i] += (internal.weights6[I][i] + delta_layer6_perExpert[expertId][I][i])
								* layer5_orig[I];
				}

				layer6_perExpert.put(expertId, layer6);
			}

		} else if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert.
			 */

			for (int expertId = 0; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
				double[] layer5 = layer5_perExpert.get(expertId);

				double[] layer6 = new double[128];
				for (int i = 0; i < 128; i++) {
					layer6[i] = internal.biases6[i];
					for (int I = 0; I < 576; I++)
						layer6[i] += internal.weights6[I][i] * layer5[I];
				}

				layer6_perExpert.put(expertId, layer6);
			}
		}

		return layer6_perExpert;
	}

	Map<Integer, double[]> layer_7(Map<Integer, double[]> layer6_perExpert, int repairedLayerId) {
		final int CURRENT_LAYER = 7;
		Map<Integer, double[]> layer7_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[] layer6_orig = layer6_perExpert.get(-1);
		double[] layer7_orig = new double[128];
		for (int i = 0; i < 128; i++)
			if (layer6_orig[i] > 0)
				layer7_orig[i] = layer6_orig[i];
			else
				layer7_orig[i] = 0;
		layer7_perExpert.put(-1, layer7_orig);

		if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert. Note: the repair cannot happen in this layer because
			 * no weights are involved.
			 */

			for (int expertId = 0; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
				double[] layer6 = layer6_perExpert.get(expertId);

				double[] layer7 = new double[128];
				for (int i = 0; i < 128; i++)
					if (layer6[i] > 0)
						layer7[i] = layer6[i];
					else
						layer7[i] = 0;

				layer7_perExpert.put(expertId, layer7);
			}
		}

		return layer7_perExpert;
	}

	Map<Integer, double[]> layer_8(Map<Integer, double[]> layer7_perExpert, int repairedLayerId) {
		final int CURRENT_LAYER = 8;
		Map<Integer, double[]> layer8_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[] layer7_orig = layer7_perExpert.get(-1);
		double[] layer8_orig = new double[10];
		for (int i = 0; i < 10; i++) {
			layer8_orig[i] = internal.biases8[i];
			for (int I = 0; I < 128; I++)
				layer8_orig[i] += internal.weights8[I][i] * layer7_orig[I];
		}
		layer8_perExpert.put(-1, layer8_orig);

		if (repairedLayerId == CURRENT_LAYER) {

			/*
			 * If the repair starts in this layer, then the layer will calculate the output
			 * with the adjusted weights for each expert.
			 */

			double[][][] delta_layer8_perExpert = (double[][][]) weight_delta;

			for (int expertId = 0; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
				double[] layer8 = new double[10];
				for (int i = 0; i < 10; i++) {
					layer8[i] = internal.biases8[i];
					for (int I = 0; I < 128; I++)
						layer8[i] += (internal.weights8[I][i] + delta_layer8_perExpert[expertId][I][i])
								* layer7_orig[I];
				}

				layer8_perExpert.put(expertId, layer8);
			}

		} else if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert.
			 */

			for (int expertId = 0; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
				double[] layer7 = layer7_perExpert.get(expertId);

				double[] layer8 = new double[10];
				for (int i = 0; i < 10; i++) {
					layer8[i] = internal.biases8[i];
					for (int I = 0; I < 128; I++)
						layer8[i] += internal.weights8[I][i] * layer7[I];
				}

				layer8_perExpert.put(expertId, layer8);
			}
		}

		return layer8_perExpert;
	}

	/**
	 * Executes the repaired network with the given input. The executions assumes
	 * that the parameter repairedLayerId specifies the repaired layer.
	 * 
	 * @param input
	 * @param repairedLayerId
	 * @return Mapping from expert network to values at the last layer.
	 * @throws IOException
	 */
	Map<Integer, double[]> run(double[][][] input, int repairedLayerId) throws IOException {

		// layer 0: conv2d_1
		Map<Integer, double[][][]> layer0_perExpert = layer_0(input, repairedLayerId);

		// layer 1: activation_1
		Map<Integer, double[][][]> layer1_perExpert = layer_1(layer0_perExpert, repairedLayerId);

		// layer 2: conv2d_2
		Map<Integer, double[][][]> layer2_perExpert = layer_2(layer1_perExpert, repairedLayerId);

		// layer 3: activation_2
		Map<Integer, double[][][]> layer3_perExpert = layer_3(layer2_perExpert, repairedLayerId);

		// layer 4: max_pooling2d_1
		Map<Integer, double[][][]> layer4_perExpert = layer_4(layer3_perExpert, repairedLayerId);

		// layer 5: flatten_1
		Map<Integer, double[]> layer5_perExpert = layer_5(layer4_perExpert, repairedLayerId);

		// layer 6: dense_1
		Map<Integer, double[]> layer6_perExpert = layer_6(layer5_perExpert, repairedLayerId);

		// layer 7: activation_3
		Map<Integer, double[]> layer7_perExpert = layer_7(layer6_perExpert, repairedLayerId);

		// layer 8: dense_2
		Map<Integer, double[]> layer8_perExpert = layer_8(layer7_perExpert, repairedLayerId);

		/*
		 * At this point we have layer8_perExpert with the original calculation results
		 * on position -1, and if there been a repaired layer, then the expert results
		 * are stored in positions 0-9.
		 */

		return layer8_perExpert;
	}

	/*
	 * *****************************************************************************
	 * Combination Implementation
	 * *****************************************************************************
	 */

	/**
	 * Enumeration for the supported expert combination methods.
	 */
	enum COMBINATION_METHOD {
		NAIVE, AVERAGE, FULL, PREC, CONF, VOTES, PVC, ORIG, ALL
	};

	/**
	 * Selects layer with maximum confidence.
	 * 
	 * @param layer8_orig - values at last layer.
	 * @return final decision for original weights
	 */
	private static int selectLabelWithMaxConfidence(double[] layer8) {
		// layer 9: activation_4
//		double[] layer9 = new double[10];
//		for (int i = 0; i < 10; i++)
//			layer9[i] = layer8[i];
//		int ret = 0;
//		double res = -100000;
//		for (int i = 0; i < 10; i++) {
//			if (layer9[i] > res) {
//				res = layer9[i];
//				ret = i;
//			}
//		}
//		return ret;

		int largest = 0;
		for (int i = 1; i < layer8.length; i++) {
			if (layer8[i] > layer8[largest]) {
				largest = i;
			}
		}
		return largest;
	}

	private static List<Integer> collectExpertClaims(int[] expertIDs, Map<Integer, double[]> result) {
		List<Integer> expertClaims = new ArrayList<>();
		for (int expertId : expertIDs) {
			int largest = selectLabelWithMaxConfidence(result.get(expertId));
			if (largest == expertId) {
				expertClaims.add(expertId);
			}
		}
		return expertClaims;
	}

	/**
	 * NAIVE Combination
	 * 
	 * Is there a unique expert who votes for itself? Otherwise return original
	 * choice.
	 * 
	 * @param expertClaims
	 * @param origLabel
	 * @param expertIDs
	 * @return
	 */
	private static int combineExpertsByNaive(List<Integer> expertClaims, int origLabel) {
		if (expertClaims.size() == 1) {
			return expertClaims.get(0);
		} else {
			return origLabel;
		}
	}

	/**
	 * PREC Combination
	 * 
	 * @param expertClaims
	 * @param origLabel
	 * @param trainPrecision
	 * @param expertIDs
	 * @return
	 */
	private static Integer combineExpertsByPrecision(List<Integer> expertClaims, int origLabel,
			double[] trainPrecision) {

		/* Check whether this no or only one expert claiming its label. */
		if (expertClaims.isEmpty()) {
			return origLabel;
		}
		if (expertClaims.size() == 1) {
			return expertClaims.get(0);
		}

		/* Select expert with highest precision on training data. */
		int maxExpertId = expertClaims.get(0);
		double maxPrecision = trainPrecision[maxExpertId];
		for (int expertId : expertClaims) {
			if (trainPrecision[expertId] > maxPrecision) {
				maxPrecision = trainPrecision[expertId];
				maxExpertId = expertId;
			}
		}
		return maxExpertId;
	}

	/**
	 * VOTES Combination
	 * 
	 * @param result
	 * @param expertClaims
	 * @param origLabel
	 * @param expertIDs
	 * @return
	 */
	private static int combineExpertsByVotes(Map<Integer, double[]> result, List<Integer> expertClaims, int origLabel,
			int[] expertIDs) {

		/* Check whether this no or only one expert claiming its label. */
		if (expertClaims.isEmpty()) {
			return origLabel;
		}
		if (expertClaims.size() == 1) {
			return expertClaims.get(0);
		}

		/* Collect the vote by each expert, i.e., the label with maximum confidence. */
		int[] votes = new int[NUMBER_OF_EXPERTS];
		for (int expertId : expertIDs) {
			int label = selectLabelWithMaxConfidence(result.get(expertId));
			votes[label]++;
		}

		/*
		 * Choose expert that votes for itself and that received most of the other
		 * votes.
		 */
		int maxVotedExpertId = expertClaims.get(0);
		for (int expertId : expertClaims) {
			if (votes[expertId] > votes[maxVotedExpertId]) {
				maxVotedExpertId = expertId;
			}
		}

		return maxVotedExpertId;
	}

	/**
	 * CONF Combination
	 * 
	 * @param expertClaims
	 * @param result
	 * @param origLabel
	 * @param expertIDs
	 * @return
	 */
	private static int combineExpertsByConfidence(Map<Integer, double[]> result, List<Integer> expertClaims,
			int origLabel) {

		/* Check whether this no or only expert claiming its label. */
		if (expertClaims.isEmpty()) {
			return origLabel;
		}
		if (expertClaims.size() == 1) {
			return expertClaims.get(0);
		}

		/* Collect expert with highest confidence in its own label. */
		int highestConfidenceId = expertClaims.get(0);
		double highestConfidenceValue = result.get(highestConfidenceId)[highestConfidenceId];
		for (int expertId : expertClaims) {
			double confidenceValue = result.get(expertId)[expertId];
			if (confidenceValue > highestConfidenceValue) {
				highestConfidenceValue = confidenceValue;
				highestConfidenceId = expertId;
			}
		}

		return highestConfidenceId;
	}

	/**
	 * PVC Combination
	 * 
	 * @param result
	 * @param expertClaims
	 * @param origLabel
	 * @param trainPrecision
	 * @param expertIDs
	 * @return
	 */
	private static Integer combineExpertsByPVC(Map<Integer, double[]> result, List<Integer> expertClaims, int origLabel,
			double[] trainPrecision, int[] expertIDs) {

		/* Check whether this no or only one expert claiming its label. */
		if (expertClaims.isEmpty()) {
			return origLabel;
		}
		if (expertClaims.size() == 1) {
			return expertClaims.get(0);
		}

		/* Initialize scores. */
		Map<Integer, Integer> scorePerExpert = new HashMap<>();
		for (int expertId : expertClaims) {
			scorePerExpert.put(expertId, 0);
		}

		/* Calculate scores */
		int labelByPrecision = combineExpertsByPrecision(expertClaims, origLabel, trainPrecision);
		scorePerExpert.put(labelByPrecision, scorePerExpert.get(labelByPrecision) + 1);

		int labelByVotes = combineExpertsByVotes(result, expertClaims, origLabel, expertIDs);
		scorePerExpert.put(labelByVotes, scorePerExpert.get(labelByVotes) + 1);

		int labelByConfidence = combineExpertsByConfidence(result, expertClaims, origLabel);
		scorePerExpert.put(labelByConfidence, scorePerExpert.get(labelByConfidence) + 1);

		/* Pick maximum score. */
		int maxScore = -1;
		int maxScoreExpert = -1;
		for (Entry<Integer, Integer> expertScorePair : scorePerExpert.entrySet()) {
			if (expertScorePair.getValue() > maxScore) {
				maxScore = expertScorePair.getValue();
				maxScoreExpert = expertScorePair.getKey();
			}
		}

		return maxScoreExpert;
	}

	private static Map<COMBINATION_METHOD, Integer> combineExperts(COMBINATION_METHOD combMethod,
			Map<Integer, double[]> result, int origLabel, double[] trainPrecision, int[] expertIDs) {
		Map<COMBINATION_METHOD, Integer> combinedResults = new HashMap<>();

		if (combMethod.equals(COMBINATION_METHOD.ORIG) || combMethod.equals(COMBINATION_METHOD.ALL)) {
			combinedResults.put(COMBINATION_METHOD.ORIG, origLabel);
		}

		List<Integer> expertClaims = collectExpertClaims(expertIDs, result);

		if (combMethod.equals(COMBINATION_METHOD.AVERAGE) || combMethod.equals(COMBINATION_METHOD.ALL)) {
			combinedResults.put(COMBINATION_METHOD.AVERAGE, selectLabelWithMaxConfidence(result.get(11)));
		}

		if (combMethod.equals(COMBINATION_METHOD.FULL) || combMethod.equals(COMBINATION_METHOD.ALL)) {
			combinedResults.put(COMBINATION_METHOD.FULL, selectLabelWithMaxConfidence(result.get(10)));
		}

		if (combMethod.equals(COMBINATION_METHOD.NAIVE) || combMethod.equals(COMBINATION_METHOD.ALL)) {
			combinedResults.put(COMBINATION_METHOD.NAIVE, combineExpertsByNaive(expertClaims, origLabel));
		}

		if (combMethod.equals(COMBINATION_METHOD.PREC) || combMethod.equals(COMBINATION_METHOD.ALL)) {
			combinedResults.put(COMBINATION_METHOD.PREC,
					combineExpertsByPrecision(expertClaims, origLabel, trainPrecision));
		}

		if (combMethod.equals(COMBINATION_METHOD.CONF) || combMethod.equals(COMBINATION_METHOD.ALL)) {
			combinedResults.put(COMBINATION_METHOD.CONF, combineExpertsByConfidence(result, expertClaims, origLabel));
		}

		if (combMethod.equals(COMBINATION_METHOD.VOTES) || combMethod.equals(COMBINATION_METHOD.ALL)) {
			combinedResults.put(COMBINATION_METHOD.VOTES,
					combineExpertsByVotes(result, expertClaims, origLabel, expertIDs));
		}

		if (combMethod.equals(COMBINATION_METHOD.PVC) || combMethod.equals(COMBINATION_METHOD.ALL)) {
			combinedResults.put(COMBINATION_METHOD.PVC,
					combineExpertsByPVC(result, expertClaims, origLabel, trainPrecision, expertIDs));
		}

		return combinedResults;
	}

	/*
	 * *****************************************************************************
	 * I/O Operations - reading weights from Z3
	 * *****************************************************************************
	 */

	public static Object loadRepairedWeights(String path, int repairedLayerId) throws IOException {

		if (repairedLayerId == 0) {

			throw new RuntimeException("Layer " + repairedLayerId + " not supported yet!"); // TODO

		} else if (repairedLayerId == 2) {

			throw new RuntimeException("Layer " + repairedLayerId + " not supported yet!"); // TODO

		} else if (repairedLayerId == 6) {
			/*
			 * 10 slots for experts, 1 slot for full repair, 1 slot for average weights of
			 * first 10 slots
			 */
			double[][][] weight_delta = new double[12][576][128];

			ArrayList<Integer> num0 = new ArrayList<Integer>();
			ArrayList<Integer> num1 = new ArrayList<Integer>();
			ArrayList<Double> num2 = new ArrayList<Double>();

			/* Read deltas for experts 0..9 */
			for (int expertId = 0; expertId < NUMBER_OF_EXPERTS; expertId++) {
				loadDeltasFromZ3File(path, expertId, num0, num1, num2);
				for (int i = 0; i < num1.size(); i++) {
					weight_delta[expertId][num1.get(i)][num0.get(i)] = num2.get(i);
					System.out.println(expertId + " : " + num1.get(i) + " : " + num0.get(i) + " -> "
							+ weight_delta[expertId][num1.get(i)][num0.get(i)]);
				}
			}

			/* Read deltas for full repair. */
			loadDeltasFromZ3File(path, NUMBER_OF_EXPERTS, num0, num1, num2);
			for (int i = 0; i < num1.size(); i++) {
				weight_delta[10][num1.get(i)][num0.get(i)] = num2.get(i);
				System.out.println(NUMBER_OF_EXPERTS + " : " + num1.get(i) + " : " + num0.get(i) + " -> "
						+ weight_delta[NUMBER_OF_EXPERTS][num1.get(i)][num0.get(i)]);
			}

			/* Calculate average deltas for experts. */
			for (int i = 0; i < 128; i++) {
				for (int I = 0; I < 576; I++) {
					double sum = 0.0;
					for (int expertId = 0; expertId < NUMBER_OF_EXPERTS; expertId++) {
						sum += weight_delta[expertId][I][i];
					}
					weight_delta[NUMBER_OF_EXPERTS + 1][I][i] = sum / NUMBER_OF_EXPERTS;
				}
			}

			return weight_delta;
		} else if (repairedLayerId == 8) {
			throw new RuntimeException("Layer " + repairedLayerId + " not supported yet!"); // TODO
		} else {
			throw new RuntimeException("Layer " + repairedLayerId + " cannot be repaired!");
		}

	}

	public static void loadDeltasFromZ3File(String path, int lab, ArrayList<Integer> num0, ArrayList<Integer> num1,
			ArrayList<Double> num2) {

		num0.clear();
		num1.clear();
		num2.clear();

		String line;
		Pattern p = Pattern.compile("[-+]?[0-9]*\\.?[0-9]+");

		String readfile;
		if (lab == 10) {
			readfile = path + "/full.txt";
		} else {
//			readfile = path + "/lowquality_label" + lab + ".txt";
			readfile = path + "/label" + lab + ".txt";
		}

		try (FileReader frread = new FileReader(readfile); BufferedReader brread = new BufferedReader(frread)) {
			line = "";
			Matcher m;
//			int count = 0;
//			Integer[] edgeslist;
//			int keys = 0;
//			Map<Integer, Integer[]> Edges = new HashMap<Integer, Integer[]>();
//			String[] r;
//			Integer[] numbers = {};
			brread.readLine();
			brread.readLine();

			while ((line = brread.readLine()) != null) {
				String a = "";
				String b = "";
				Double c = 0.0;
				int countnumbers = 0;
				// System.out.println("=>"+line);
				if (line.contains("define")) {
					line = line.replaceAll("_", " ");
					String[] nums = line.replaceAll("[^0-9 ]", "").trim().split(" +");
					// System.out.println(nums[1]);
					num0.add(Integer.valueOf(nums[0]));
					num1.add(Integer.valueOf(nums[1]));
				} else if (line.contains("/") && line.chars().filter(ch -> ch == '.').count() == 2) {
					m = p.matcher(line);
					while (m.find()) {
						countnumbers++;
						if (countnumbers == 1) {
							a = m.group();
						} else if (countnumbers == 2) {
							b = m.group();
						}
					}
					c = Double.valueOf(a) / Double.valueOf(b);
					if (line.contains("-")) {
						c = c * -1;
					}
					num2.add(c);
				} else if (line.contains("/") && line.chars().filter(ch -> ch == '.').count() == 1) {
					m = p.matcher(line);
					while (m.find()) {
						a = m.group();
					}
					if (line.contains("-")) {
						c = -1.0;
					} else {
						c = 1.0;
					}
					line = brread.readLine();
					m = p.matcher(line);
					while (m.find()) {
						b = m.group();
					}

					c = c * Double.valueOf(a) / Double.valueOf(b);

					num2.add(c);
				} else {
					m = p.matcher(line);
					while (m.find()) {
						c = Double.valueOf(m.group());
					}
					if (line.contains("-")) {
						c = c * -1;
					}
					num2.add(c);
				}

			}
		} catch (IOException e) {
			throw new RuntimeException("Error during z3 output parsing.", e);
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
	 * Experiment Execution - Testing
	 * *****************************************************************************
	 */

	/**
	 * Enumeration for supported models.
	 */
	enum SUBJECT {

		// LOW_QUALITY(workspace + "data\\low-quality\\"),
		// LOW_QUALITY("/Users/yannic/repositories/DNN-Repair/experiments/pattern-based-repair/"),
		LOW_QUALITY("/Users/yannic/repositories/DNN-Repair/experiments/pattern-based-repair/low-quality/"),

		POISONED(workspace + "data\\poisoned\\"), HIGH_QUALITY(workspace + "data\\high-quality\\");

		private String path;

		SUBJECT(String path) {
			this.path = path;
		}

		public String getPath() {
			return path;
		}
	}

	static String workspace = "C:\\Users\\dgopinat\\eclipse-workspace\\mnist_example\\mnist_example\\";

	public static void runExperiment() throws NumberFormatException, IOException {

		/* ************************************************************** */
		/* Parameters for the experiment. */

		SUBJECT subject = SUBJECT.LOW_QUALITY;

		COMBINATION_METHOD combMethod = COMBINATION_METHOD.ALL;

		int repairedLayerId = 6; // {0 | 2 | 6 | 8}

		String labelFile = subject.getPath() + "mnist_test_labels.txt";
		String inputFile = subject.getPath() + "mnist_test.txt";

		int stopAfter = 10000; // 60000;

		// precision of the expert on the train set, used for PVC or PREC combination.
		double[] trainPrecision = new double[] { 0.971, 0.9849, 0.6278, 0.9502, 0.677, 0.9135, 0.99, 0.9571, 0.9748,
				0.9312 };

		// Experts whose F1 score on train set is more than original model.
		boolean useF1Selection = false;
		int[] f1SelectedExperts = new int[] { 6, 8, 9 };

		/* ************************************************************** */
		
		

		System.out.println("PATH:" + subject.getPath());

		InternalData data = new InternalData(subject.getPath(), "weights0.txt", "weights2.txt", "weights6.txt",
				"weights8.txt", "biases0.txt", "biases2.txt", "biases6.txt", "biases8.txt");
		Object repaired_weight_deltas = loadRepairedWeights(subject.getPath(), repairedLayerId);
		DNNtGeneralCombination model = new DNNtGeneralCombination(data, repaired_weight_deltas);

		/* Initialize analytics */
		Map<Object, Integer> passCounter = new HashMap<>();
		Map<Object, Integer> failCounter = new HashMap<>();
		for (COMBINATION_METHOD x : COMBINATION_METHOD.values()) {
			passCounter.put(x, 0);
			failCounter.put(x, 0);
		}
		for (int x = 0; x < NUMBER_OF_EXPERTS + 2; x++) {
			passCounter.put(x, 0);
			failCounter.put(x, 0);
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
			expertIDs = new int[NUMBER_OF_EXPERTS];
			for (int i = 0; i < NUMBER_OF_EXPERTS; i++) {
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

			Map<Integer, double[]> result = model.run(input, repairedLayerId);

			// Extract original decision.
			int origLabel = selectLabelWithMaxConfidence(result.get(-1)); /* ORIG */

			// Determine final decisions by experts.
			Map<COMBINATION_METHOD, Integer> results = combineExperts(combMethod, result, origLabel, trainPrecision,
					expertIDs);

			// Print results and collect analytics.
			System.out.print(count + "; IDEAL: " + labels[count] + "; ");
			for (Entry<COMBINATION_METHOD, Integer> combinedResult : results.entrySet()) {
				COMBINATION_METHOD currentCombinationMethod = combinedResult.getKey();
				int label = combinedResult.getValue();

				boolean passed = (label == labels[count]);
				if (passed) {
					passCounter.put(currentCombinationMethod, passCounter.get(currentCombinationMethod) + 1);
				} else {
					failCounter.put(currentCombinationMethod, failCounter.get(currentCombinationMethod) + 1);
				}

				System.out.print(currentCombinationMethod + ": " + (passed ? "PASS" : "FAIL") + " " + label + "; ");
			}

			// Results of single experts are only for targeted repair interesting.
			int expertId = labels[count]; // only pick local expert
			int label = selectLabelWithMaxConfidence(result.get(expertId));
			boolean passed = (label == labels[count]);
			if (passed) {
				passCounter.put(expertId, passCounter.get(expertId) + 1);
			} else {
				failCounter.put(expertId, failCounter.get(expertId) + 1);
			}
			System.out.print("ExpertL" + expertId + ": " + (passed ? "PASS" : "FAIL") + " " + label + "; ");

			System.out.println();
			count++;

			if (count == stopAfter) {
				break;
			}

		}

		br.close();

		// Calculate and print accuracy.
		System.out.println();
		System.out.println("COMBINATION;ACCURACY;PASS;FAIL");
		if (combMethod.equals(COMBINATION_METHOD.ALL)) {
			for (COMBINATION_METHOD combinationMethod : COMBINATION_METHOD.values()) {
				if (combinationMethod.equals(COMBINATION_METHOD.ALL)) {
					continue;
				}
				int pass = passCounter.get(combinationMethod);
				int fail = failCounter.get(combinationMethod);
				double accuracy = round((((double) pass) / (pass + fail)) * 100.0, 2);

				System.out.println(combinationMethod + ";" + accuracy + ";" + pass + ";" + fail);
			}
		} else {
			int pass = passCounter.get(combMethod);
			int fail = failCounter.get(combMethod);
			double accuracy = round((((double) pass) / (pass + fail)) * 100.0, 2);

			System.out.println(combMethod + ";" + accuracy + ";" + pass + ";" + fail);
		}
		for (int expertId = 0; expertId < NUMBER_OF_EXPERTS; expertId++) {
			int pass = passCounter.get(expertId);
			int fail = failCounter.get(expertId);
			double accuracy = round((((double) pass) / (pass + fail)) * 100.0, 2);

			System.out.println("L" + expertId + ";" + accuracy + ";" + pass + ";" + fail);
		}
		System.out.println();

	}

	public static void main(String[] args) {
		try {
			runExperiment();
		} catch (NumberFormatException | IOException e) {
			e.printStackTrace();
		}
	}

}
