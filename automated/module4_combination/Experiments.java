import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Experiment Execution - Testing
 */
public class Experiments {

	/**
	 * Enumeration for supported models.
	 */
	enum SUBJECT {

		////////////////////////////////////////////////////////////////////////////////////////////////////

		LOW_QUALITY_PATTERN_TEST("/Users/yannic/experiments/nnrepair/mnist_low_quality", "/divya/layer6",
				"label", 6, "/mnist_test_labels.txt", "/mnist_test.txt", false,
				new double[] { 0.970734126984127, 0.9848417298261257, 0.6285016977928692, 0.950096587250483,
						0.6775396085740913, 0.914295509084676, 0.9900068917987594, 0.9570011025358324,
						0.9746994437466355, 0.9322475570032573 },
				new int[] { 6, 8, 9 }),

		LOW_QUALITY_PATTERN_TRAINING("/Users/yannic/experiments/nnrepair/mnist_low_quality", "/divya/layer6",
				"label", 6, "/mnist_train_labels.txt", "/mnist_train.txt", false,
				new double[] { 0.970734126984127, 0.9848417298261257, 0.6285016977928692, 0.950096587250483,
						0.6775396085740913, 0.914295509084676, 0.9900068917987594, 0.9570011025358324,
						0.9746994437466355, 0.9322475570032573 },
				new int[] { 6, 8, 9 }),

		////////////////////////////////////////////////////////////////////////////////////////////////////

		LOW_QUALITY_LAST_LAYER_TEST("/Users/yannic/experiments/nnrepair/mnist_low_quality", "/usman/layer8",
				"label", 8, "/mnist_test_labels.txt", "/mnist_test.txt", false,
				new double[] { 0.9346001583531275, 0.7040612865988036, 0.7582460225067909, 0.3880053650124545,
						0.7409247229652274, 0.32622190080154284, 0.8278192313107138, 0.9191318566968931,
						0.8989931276969794, 0.4081175647305808 },
				new int[] {}),

		LOW_QUALITY_LAST_LAYER_TRAINING("/Users/yannic/experiments/nnrepair/mnist_low_quality", "/usman/layer8",
				"label", 8, "/mnist_train_labels.txt", "/mnist_train.txt", false,
				new double[] { 0.9346001583531275, 0.7040612865988036, 0.7582460225067909, 0.3880053650124545,
						0.7409247229652274, 0.32622190080154284, 0.8278192313107138, 0.9191318566968931,
						0.8989931276969794, 0.4081175647305808 },
				new int[] {}),

		LOW_QUALITY_LAST_LAYER_ExpA_TEST("/Users/yannic/experiments/nnrepair/mnist_low_quality", "/usman/ExpA",
				"label", 8, "/mnist_test_labels.txt", "/mnist_test.txt", false,
				new double[] { 0.9544933704370601, 0.7040612865988036, 0.2524199553239017, 0.33208040529498284,
						0.39791652388458637, 0.3175711352302728, 0.8814053390610617, 0.8993692239988265,
						0.8386951292844257, 0.4046511627906977 },
				new int[] {}),

		LOW_QUALITY_LAST_LAYER_ExpA_TRAINING("/Users/yannic/experiments/nnrepair/mnist_low_quality",
				"/usman/ExpA", "label", 8, "/mnist_train_labels.txt", "/mnist_train.txt", false,
				new double[] { 0.9544933704370601, 0.7040612865988036, 0.2524199553239017, 0.33208040529498284,
						0.39791652388458637, 0.3175711352302728, 0.8814053390610617, 0.8993692239988265,
						0.8386951292844257, 0.4046511627906977 },
				new int[] {}),

		LOW_QUALITY_LAST_LAYER_ExpB_TEST("/Users/yannic/experiments/nnrepair/mnist_low_quality", "/usman/ExpB",
				"label", 8, "/mnist_test_labels.txt", "/mnist_test.txt", false,
				new double[] { 0.9156458365638084, 0.9106679415380412, 0.562788906009245, 0.6444893187373791,
						0.5581665216129968, 0.6281117895725693, 0.9350566459230892, 0.9185229481237853,
						0.8386538763956597, 0.5247171996080876 },
				new int[] {}),

		LOW_QUALITY_LAST_LAYER_ExpB_TRAINING("/Users/yannic/experiments/nnrepair/mnist_low_quality",
				"/usman/ExpB", "label", 8, "/mnist_train_labels.txt", "/mnist_train.txt", false,
				new double[] { 0.9156458365638084, 0.9106679415380412, 0.562788906009245, 0.6444893187373791,
						0.5581665216129968, 0.6281117895725693, 0.9350566459230892, 0.9185229481237853,
						0.8386538763956597, 0.5247171996080876 },
				new int[] {}),

		LOW_QUALITY_LAST_LAYER_ExpC_TEST("/Users/yannic/experiments/nnrepair/mnist_low_quality", "/usman/ExpC",
				"label", 8, "/mnist_test_labels.txt", "/mnist_test.txt", false,
				new double[] { 0.841407808492448, 0.9379556382813712, 0.9494325767690254, 0.6368750663411528,
						0.7353535353535353, 0.7935308343409916, 0.8278192313107138, 0.9191318566968931,
						0.9010673888800382, 0.7257415786827551 },
				new int[] {}),

		LOW_QUALITY_LAST_LAYER_ExpC_TRAINING("/Users/yannic/experiments/nnrepair/mnist_low_quality",
				"/usman/ExpC", "label", 8, "/mnist_train_labels.txt", "/mnist_train.txt", false,
				new double[] { 0.841407808492448, 0.9379556382813712, 0.9494325767690254, 0.6368750663411528,
						0.7353535353535353, 0.7935308343409916, 0.8278192313107138, 0.9191318566968931,
						0.9010673888800382, 0.7257415786827551 },
				new int[] {}),

		LOW_QUALITY_LAST_LAYER_ExpD_TEST("/Users/yannic/experiments/nnrepair/mnist_low_quality", "/usman/ExpD",
				"label", 8, "/mnist_test_labels.txt", "/mnist_test.txt", false,
				new double[] { 0.9658736669401149, 0.9321693448702101, 0.8903286978508217, 0.9058468755235383,
						0.6448184233835252, 0.826530612244898, 0.9033451518421458, 0.9191318566968931,
						0.9221577930065015, 0.7257415786827551 },
				new int[] {}),

		LOW_QUALITY_LAST_LAYER_ExpD_TRAINING("/Users/yannic/experiments/nnrepair/mnist_low_quality",
				"/usman/ExpD", "label", 8, "/mnist_train_labels.txt", "/mnist_train.txt", false,
				new double[] { 0.9658736669401149, 0.9321693448702101, 0.8903286978508217, 0.9058468755235383,
						0.6448184233835252, 0.826530612244898, 0.9033451518421458, 0.9191318566968931,
						0.9221577930065015, 0.7257415786827551 },
				new int[] {}),

		////////////////////////////////////////////////////////////////////////////////////////////////////

		POISONED_LAST_LAYER_ExpA_TEST("/Users/yannic/experiments/nnrepair/mnist_poisoned", "/usman/ExpA",
				"label", 8, "/mnist_test_labels.txt", "/mnist_test.txt", false,
				new double[] { 0.9564160725858717, 0.8183695784230348, 0.9087283325663446, 0.8839259901705695,
						0.9368625546381739, 0.7205705905879216, 0.9756622516556291, 0.917679800790977,
						0.9291553133514986, 0.896787943370376 },
				new int[] {}),

		POISONED_LAST_LAYER_ExpA_POISONED_TEST("/Users/yannic/experiments/nnrepair/mnist_poisoned",
				"/usman/ExpA", "label", 8, "/poisoned_mnist_test_label_csv.txt", "/poisoned_mnist_test_csv.txt", false,
				new double[] { 0.9564160725858717, 0.8183695784230348, 0.9087283325663446, 0.8839259901705695,
						0.9368625546381739, 0.7205705905879216, 0.9756622516556291, 0.917679800790977,
						0.9291553133514986, 0.896787943370376 },
				new int[] {}),

		POISONED_LAST_LAYER_ExpA_TRAINING("/Users/yannic/experiments/nnrepair/mnist_poisoned", "/usman/ExpA",
				"label", 8, "/poisoned_mnist_train_label_csv.txt", "/poisoned_mnist_train_csv.txt", false,
				new double[] { 0.9564160725858717, 0.8183695784230348, 0.9087283325663446, 0.8839259901705695,
						0.9368625546381739, 0.7205705905879216, 0.9756622516556291, 0.917679800790977,
						0.9291553133514986, 0.896787943370376 },
				new int[] {}),

		POISONED_LAST_LAYER_ExpB_TEST("/Users/yannic/experiments/nnrepair/mnist_poisoned", "/usman/ExpB",
				"label", 8, "/mnist_test_labels.txt", "/mnist_test.txt", false,
				new double[] { 0.7458417338709677, 0.8056917374148033, 0.9329767149150409, 0.8909462020702726,
						0.9306069876026405, 0.8545224541429475, 0.8876844323998796, 0.917679800790977,
						0.9247294716740929, 0.8858214553638409 },
				new int[] {}),

		POISONED_LAST_LAYER_ExpB_POISONED_TEST("/Users/yannic/experiments/nnrepair/mnist_poisoned",
				"/usman/ExpB", "label", 8, "/poisoned_mnist_test_label_csv.txt", "/poisoned_mnist_test_csv.txt", false,
				new double[] { 0.7458417338709677, 0.8056917374148033, 0.9329767149150409, 0.8909462020702726,
						0.9306069876026405, 0.8545224541429475, 0.8876844323998796, 0.917679800790977,
						0.9247294716740929, 0.8858214553638409 },
				new int[] {}),

		POISONED_LAST_LAYER_ExpB_TRAINING("/Users/yannic/experiments/nnrepair/mnist_poisoned", "/usman/ExpB",
				"label", 8, "/poisoned_mnist_train_label_csv.txt", "/poisoned_mnist_train_csv.txt", false,
				new double[] { 0.7458417338709677, 0.8056917374148033, 0.9329767149150409, 0.8909462020702726,
						0.9306069876026405, 0.8545224541429475, 0.8876844323998796, 0.917679800790977,
						0.9247294716740929, 0.8858214553638409 },
				new int[] {}),

		POISONED_LAST_LAYER_ExpC_TEST("/Users/yannic/experiments/nnrepair/mnist_poisoned", "/usman/ExpC",
				"label", 8, "/mnist_test_labels.txt", "/mnist_test.txt", false,
				new double[] { 0.9287172240540116, 0.9037017167381974, 0.8927980754773718, 0.8785570566254671,
						0.9361943319838056, 0.8261733679865464, 0.9714944801449992, 0.917679800790977,
						0.9285714285714286, 0.8858214553638409 },
				new int[] {}),

		POISONED_LAST_LAYER_ExpC_POISONED_TEST("/Users/yannic/experiments/nnrepair/mnist_poisoned",
				"/usman/ExpC", "label", 8, "/poisoned_mnist_test_label_csv.txt", "/poisoned_mnist_test_csv.txt", false,
				new double[] { 0.9287172240540116, 0.9037017167381974, 0.8927980754773718, 0.8785570566254671,
						0.9361943319838056, 0.8261733679865464, 0.9714944801449992, 0.917679800790977,
						0.9285714285714286, 0.8858214553638409 },
				new int[] {}),

		POISONED_LAST_LAYER_ExpC_TRAINING("/Users/yannic/experiments/nnrepair/mnist_poisoned", "/usman/ExpC",
				"label", 8, "/poisoned_mnist_train_label_csv.txt", "/poisoned_mnist_train_csv.txt", false,
				new double[] { 0.9287172240540116, 0.9037017167381974, 0.8927980754773718, 0.8785570566254671,
						0.9361943319838056, 0.8261733679865464, 0.9714944801449992, 0.917679800790977,
						0.9285714285714286, 0.8858214553638409 },
				new int[] {}),

		POISONED_LAST_LAYER_ExpD_TEST("/Users/yannic/experiments/nnrepair/mnist_poisoned", "/usman/ExpD",
				"label", 8, "/mnist_test_labels.txt", "/mnist_test.txt", false,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] { 7 }),

		POISONED_LAST_LAYER_ExpD_POISONED_TEST("/Users/yannic/experiments/nnrepair/mnist_poisoned",
				"/usman/ExpD", "label", 8, "/poisoned_mnist_test_label_csv.txt", "/poisoned_mnist_test_csv.txt", false,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] { 7 }),

		POISONED_LAST_LAYER_ExpD_TRAINING("/Users/yannic/experiments/nnrepair/mnist_poisoned", "/usman/ExpD",
				"label", 8, "/poisoned_mnist_train_label_csv.txt", "/poisoned_mnist_train_csv.txt", false,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] { 7 }),

		////////////////////////////////////////////////////////////////////////////////////////////////////

		ADVERSARIAL_LAST_LAYER_Eps0_1_ExpA_TEST("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.1_ExpA", "solution", 8, "/mnist_test_label_csv.txt", "/mnist_test_csv_fgsm_epsilon0.1.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),

		ADVERSARIAL_LAST_LAYER_Eps0_1_ExpA_TRAINING("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.1_ExpA", "solution", 8, "/mnist_adv_val_label.txt", "/mnist_val_csv_fgsm_epsilon0.1.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),
		
		ADVERSARIAL_LAST_LAYER_Eps0_1_ExpB_TEST("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.1_ExpB", "solution", 8, "/mnist_test_label_csv.txt", "/mnist_test_csv_fgsm_epsilon0.1.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),

		ADVERSARIAL_LAST_LAYER_Eps0_1_ExpB_TRAINING("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.1_ExpB", "solution", 8, "/mnist_adv_val_label.txt", "/mnist_val_csv_fgsm_epsilon0.1.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),
		
		ADVERSARIAL_LAST_LAYER_Eps0_1_ExpC_TEST("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.1_ExpC", "solution", 8, "/mnist_test_label_csv.txt", "/mnist_test_csv_fgsm_epsilon0.1.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),

		ADVERSARIAL_LAST_LAYER_Eps0_1_ExpC_TRAINING("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.1_ExpC", "solution", 8, "/mnist_adv_val_label.txt", "/mnist_val_csv_fgsm_epsilon0.1.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),
		
		ADVERSARIAL_LAST_LAYER_Eps0_1_ExpD_TEST("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.1_ExpD", "solution", 8, "/mnist_test_label_csv.txt", "/mnist_test_csv_fgsm_epsilon0.1.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),

		ADVERSARIAL_LAST_LAYER_Eps0_1_ExpD_TRAINING("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.1_ExpD", "solution", 8, "/mnist_adv_val_label.txt", "/mnist_val_csv_fgsm_epsilon0.1.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),

		////////////////////////////////////////////////////////////////////////////////////////////////////
		
		ADVERSARIAL_LAST_LAYER_Eps0_05_ExpA_TEST("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.05_ExpA", "solution", 8, "/mnist_test_label_csv.txt", "/mnist_test_csv_fgsm_epsilon0.05.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),

		ADVERSARIAL_LAST_LAYER_Eps0_05_ExpA_TRAINING("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.05_ExpA", "solution", 8, "/mnist_adv_val_label.txt", "/mnist_val_csv_fgsm_epsilon0.05.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),
		
		ADVERSARIAL_LAST_LAYER_Eps0_05_ExpB_TEST("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.05_ExpB", "solution", 8, "/mnist_test_label_csv.txt", "/mnist_test_csv_fgsm_epsilon0.05.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),

		ADVERSARIAL_LAST_LAYER_Eps0_05_ExpB_TRAINING("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.05_ExpB", "solution", 8, "/mnist_adv_val_label.txt", "/mnist_val_csv_fgsm_epsilon0.05.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),
		
		ADVERSARIAL_LAST_LAYER_Eps0_05_ExpC_TEST("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.05_ExpC", "solution", 8, "/mnist_test_label_csv.txt", "/mnist_test_csv_fgsm_epsilon0.05.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),

		ADVERSARIAL_LAST_LAYER_Eps0_05_ExpC_TRAINING("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.05_ExpC", "solution", 8, "/mnist_adv_val_label.txt", "/mnist_val_csv_fgsm_epsilon0.05.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),
		
		ADVERSARIAL_LAST_LAYER_Eps0_05_ExpD_TEST("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.05_ExpD", "solution", 8, "/mnist_test_label_csv.txt", "/mnist_test_csv_fgsm_epsilon0.05.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),

		ADVERSARIAL_LAST_LAYER_Eps0_05_ExpD_TRAINING("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.05_ExpD", "solution", 8, "/mnist_adv_val_label.txt", "/mnist_val_csv_fgsm_epsilon0.05.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),
		
		////////////////////////////////////////////////////////////////////////////////////////////////////
		
		ADVERSARIAL_LAST_LAYER_Eps0_01_ExpA_TEST("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.01_ExpA", "solution", 8, "/mnist_test_label_csv.txt", "/mnist_test_csv_fgsm_epsilon0.01.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),

		ADVERSARIAL_LAST_LAYER_Eps0_01_ExpA_TRAINING("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.01_ExpA", "solution", 8, "/mnist_adv_val_label.txt", "/mnist_val_csv_fgsm_epsilon0.01.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),
		
		ADVERSARIAL_LAST_LAYER_Eps0_01_ExpB_TEST("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.01_ExpB", "solution", 8, "/mnist_test_label_csv.txt", "/mnist_test_csv_fgsm_epsilon0.01.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),

		ADVERSARIAL_LAST_LAYER_Eps0_01_ExpB_TRAINING("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.01_ExpB", "solution", 8, "/mnist_adv_val_label.txt", "/mnist_val_csv_fgsm_epsilon0.01.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),
		
		ADVERSARIAL_LAST_LAYER_Eps0_01_ExpC_TEST("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.01_ExpC", "solution", 8, "/mnist_test_label_csv.txt", "/mnist_test_csv_fgsm_epsilon0.01.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),

		ADVERSARIAL_LAST_LAYER_Eps0_01_ExpC_TRAINING("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.01_ExpC", "solution", 8, "/mnist_adv_val_label.txt", "/mnist_val_csv_fgsm_epsilon0.01.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),
		
		ADVERSARIAL_LAST_LAYER_Eps0_01_ExpD_TEST("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.01_ExpD", "solution", 8, "/mnist_test_label_csv.txt", "/mnist_test_csv_fgsm_epsilon0.01.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),

		ADVERSARIAL_LAST_LAYER_Eps0_01_ExpD_TRAINING("/Users/yannic/experiments/nnrepair/mnist_adv",
				"/usman/eps.01_ExpD", "solution", 8, "/mnist_adv_val_label.txt", "/mnist_val_csv_fgsm_epsilon0.01.txt",
				true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),
		
		////////////////////////////////////////////////////////////////////////////////////////////////////

		CIFAR_LAST_LAYER_ORIGINAL_TEST("/Users/yannic/experiments/nnrepair/cifar", "", "label", -1,
				"/cifar_test_label_csv.txt", "/cifar_test_csv.txt", true, new double[] {}, new int[] {}),

		CIFAR_LAST_LAYER_ORIGINAL_TRAINING("/Users/yannic/experiments/nnrepair/cifar", "", "label", -1,
				"/cifar_train_label_csv.txt", "/cifar_train_csv.txt", true, new double[] {}, new int[] {}),

		CIFAR_LAST_LAYER_TEST("/Users/yannic/experiments/nnrepair/cifar", "/usman/ExpD", "label", 13,
				"/cifar_test_label_csv.txt", "/cifar_test_csv.txt", true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {}),

		CIFAR_LAST_LAYER_TRAINING("/Users/yannic/experiments/nnrepair/cifar", "/usman/ExpD", "label", 13,
				"/cifar_train_label_csv.txt", "/cifar_train_csv.txt", true,
				new double[] { 0.9883939238777949, 0.9479328347678848, 0.9302873292510598, 0.8875980249782167,
						0.9655978623914495, 0.8755776142392606, 0.9721260102259608, 0.9180832356389215,
						0.9499427449697366, 0.8569770815201625 },
				new int[] {});

		////////////////////////////////////////////////////////////////////////////////////////////////////

		private String projectPath;
		private String repairPath;
		private String solutionFileNamePrefix;
		private int repairedLayerId;
		String inputFilePath;
		String labelFilePath;
		boolean needsNormalization;
		double[] trainPrecision;
		int[] f1SelectedExperts;

		SUBJECT(String projectPath, String repairPath, String solutionFileNamePrefix, int repairedLayerId,
				String labelFilePath, String inputFilePath, boolean needsNormalization, double[] trainPrecision,
				int[] f1SelectedExperts) {
			this.projectPath = projectPath;
			this.repairPath = projectPath + repairPath;
			this.solutionFileNamePrefix = solutionFileNamePrefix;
			this.repairedLayerId = repairedLayerId;
			this.labelFilePath = projectPath + labelFilePath;
			this.inputFilePath = projectPath + inputFilePath;
			this.needsNormalization = needsNormalization;
			this.trainPrecision = trainPrecision;
			this.f1SelectedExperts = f1SelectedExperts;
		}

		public String getProjectPath() {
			return projectPath;
		}

		public String getRepairPath() {
			return repairPath;
		}

		public String getSolutionFileNamePrefix() {
			return solutionFileNamePrefix;
		}

		public int getRepairedLayerId() {
			return repairedLayerId;
		}

		public String getLabelFilePath() {
			return labelFilePath;
		}

		public boolean needsNormalization() {
			return needsNormalization;
		}

		public String getInputFilePath() {
			return inputFilePath;
		}

		public double[] getTrainPrecision() {
			return trainPrecision;
		}

		public int[] getF1SelectedExperts() {
			return f1SelectedExperts;
		}
	}

	/*
	 * *****************************************************************************
	 * Utilities
	 * *****************************************************************************
	 */

	private static double round(double value, int places) {
		if (Double.isNaN(value)) {
			return value;
		}
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

	public static void runMNIST0Experiment(SUBJECT subject, ExpertCombination.COMBINATION_METHOD combMethod)
			throws NumberFormatException, IOException {
		runMNIST0Experiment(subject, combMethod, null, false);
	}

	public static void runMNIST0Experiment(SUBJECT subject, ExpertCombination.COMBINATION_METHOD combMethod,
			Integer stopAfter, boolean useF1Selection) throws NumberFormatException, IOException {

		int repairedLayerId = subject.getRepairedLayerId(); // {0 | 2 | 6 | 8}

		System.out.println("PATH:" + subject.getProjectPath());

		/* Prepare the experts. */
		int[] expertIDs;
		if (useF1Selection) {
			expertIDs = subject.getF1SelectedExperts();
		} else {
			expertIDs = new int[MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS];
			for (int i = 0; i < MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS; i++) {
				expertIDs[i] = i;
			}
		}

		MNIST0_InternalData data = new MNIST0_InternalData(subject.getProjectPath(), "weights0.txt", "weights2.txt",
				"weights6.txt", "weights8.txt", "biases0.txt", "biases2.txt", "biases6.txt", "biases8.txt");
		Object repaired_weight_deltas = Z3SolutionParsing.loadRepairedWeights_MNIST0(subject.getRepairPath(),
				subject.getSolutionFileNamePrefix(), repairedLayerId, expertIDs,
				MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS);
		MNIST0_DNNt_Combined model = new MNIST0_DNNt_Combined(data, repaired_weight_deltas);

		/* Initialize analytics */
		Map<Object, Integer> passCounter = new HashMap<>();
		Map<Object, Integer> failCounter = new HashMap<>();
		Map<Object, Integer> targetedPassCounter = new HashMap<>();
		Map<Object, Integer> targetedFailCounter = new HashMap<>();
		Map<Object, Integer> TPCounter = new HashMap<>();
		Map<Object, Integer> TNCounter = new HashMap<>();
		Map<Object, Integer> FPCounter = new HashMap<>();
		Map<Object, Integer> FNCounter = new HashMap<>();
		for (ExpertCombination.COMBINATION_METHOD x : ExpertCombination.COMBINATION_METHOD.values()) {
			passCounter.put(x, 0);
			failCounter.put(x, 0);
		}
		for (int x = 0; x < MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS + 2; x++) {
			passCounter.put(x, 0);
			failCounter.put(x, 0);
			targetedPassCounter.put(x, 0);
			targetedFailCounter.put(x, 0);
			TPCounter.put(x, 0);
			TNCounter.put(x, 0);
			FPCounter.put(x, 0);
			FNCounter.put(x, 0);
			String id = "ORIG_L" + x;
			targetedPassCounter.put(id, 0);
			targetedFailCounter.put(id, 0);
			TPCounter.put(id, 0);
			TNCounter.put(id, 0);
			FPCounter.put(id, 0);
			FNCounter.put(id, 0);
		}

		/* Read correct labels. */
		File file = new File(subject.getLabelFilePath());
		BufferedReader br = new BufferedReader(new FileReader(file));
		String st;
		Integer[] labels = new Integer[60000];
		int index = 0;
		while ((st = br.readLine()) != null) {
			labels[index] = Integer.valueOf(st);
			index++;
			if (stopAfter != null && index == stopAfter) {
				break;
			}
		}
		br.close();

		/* Read input files and execute model. */
		file = new File(subject.getInputFilePath());
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
							if (subject.needsNormalization()) {
								input[i][j][k] = (double) (val / 255.0);
							} else {
								input[i][j][k] = (double) val;
							}
						}
			}

			Map<Integer, double[]> result = model.run(input, repairedLayerId, expertIDs);

			int correctLabel = labels[count];

			// Extract original decision.
			int origLabel = ExpertCombination.selectLabelWithMaxConfidence(result.get(-1)); /* ORIG */

			// Determine final decisions by experts.
			Map<ExpertCombination.COMBINATION_METHOD, Integer> results = ExpertCombination.combineExperts(combMethod,
					result, origLabel, subject.getTrainPrecision(), expertIDs, false,
					MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS);

			// Print results and collect analytics.
			System.out.print(count + "; IDEAL: " + correctLabel + "; ");
			for (Entry<ExpertCombination.COMBINATION_METHOD, Integer> combinedResult : results.entrySet()) {
				ExpertCombination.COMBINATION_METHOD currentCombinationMethod = combinedResult.getKey();
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
			for (int expertId : expertIDs) {
				int label = ExpertCombination.selectLabelWithMaxConfidence(result.get(expertId));
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

			for (int expertId : expertIDs) {
				// Also collect results for original model specific for labels.
				boolean passed = origLabel == correctLabel;
				String id = "ORIG_L" + expertId;
				if (passed) {
					if (correctLabel == expertId) {
						TPCounter.put(id, TPCounter.get(id) + 1);
						targetedPassCounter.put(id, targetedPassCounter.get(id) + 1);
						System.out.print(id + ": " + "PASS" + " " + origLabel + "; ");
					} else {
						TNCounter.put(id, TNCounter.get(id) + 1);
						System.out.print(id + ": " + "PASS" + " " + origLabel + "; ");
					}

				} else {
					if (correctLabel == expertId) {
						FNCounter.put(id, FNCounter.get(id) + 1);
						targetedFailCounter.put(id, targetedFailCounter.get(id) + 1);
						System.out.print(id + ": " + "FAIL" + " " + origLabel + "; ");
					} else if (origLabel == expertId) {
						FPCounter.put(id, FPCounter.get(id) + 1);
						System.out.print(id + ": " + "FAIL" + " " + origLabel + "; ");
					} else {
						TNCounter.put(id, TNCounter.get(id) + 1);
						System.out.print(id + ": " + "PASS" + " " + origLabel + "; ");
					}
				}
			}

			System.out.println();
			count++;

			if (stopAfter != null && count == stopAfter) {
				break;
			}

		}

		br.close();

		// Calculate and print accuracy.
		System.out.println();
		System.out.println("COMBINATION;ACCURACY;PASS;FAIL;TAR-ACC;TAR-PASS;TAR-FAIL;TP;TN;FP;FN;PREC;RECALL;F1");
		if (combMethod.equals(ExpertCombination.COMBINATION_METHOD.ALL)) {
			for (ExpertCombination.COMBINATION_METHOD combinationMethod : ExpertCombination.COMBINATION_METHOD
					.values()) {
				if (combinationMethod.equals(ExpertCombination.COMBINATION_METHOD.ALL)) {
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
		
		double[] prec = new double[MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS];
		List<Integer> f1Experts = new ArrayList<>();
		StringBuilder bs = new StringBuilder();
		
		for (int expertId : expertIDs) {
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

			String id = "ORIG_L" + expertId;

			int targetedPass_O = targetedPassCounter.get(id);
			int targetedFail_O = targetedFailCounter.get(id);
			double targetedAccuracy_O = round((((double) targetedPass_O) / (targetedPass_O + targetedFail_O)) * 100.0,
					2);

			int TP_O = TPCounter.get(id);
			int TN_O = TNCounter.get(id);
			int FP_O = FPCounter.get(id);
			int FN_O = FNCounter.get(id);

			double precision_O = ((double) TP_O) / (TP_O + FP_O);
			double recall_O = ((double) TP_O) / (TP_O + FN_O);
			double f1_O = 2 * precision_O * recall_O / (precision_O + recall_O);

			bs.append(id + ";;;;" + targetedAccuracy_O + ";" + targetedPass_O + ";" + targetedFail_O + ";"
					+ TP_O + ";" + TN_O + ";" + FP_O + ";" + FN_O + ";" + round(precision_O * 100.0, 2) + ";"
					+ round(recall_O * 100.0, 2) + ";" + round(f1_O * 100.0, 2) + "\n");

			if (f1 > f1_O) {
				f1Experts.add(expertId);
			}
		}
		
		System.out.println(bs.toString());

		System.out.println();
		System.out.println("prec=" + Arrays.toString(prec));
		System.out.println("f1Experts=" + Arrays.toString(f1Experts.toArray()));

	}

	public static void runMNIST0CombinationOverheadExperiment(SUBJECT subject, Integer stopAfter,
			boolean useF1Selection, int[] f1SelectedExperts) throws NumberFormatException, IOException {

		int repairedLayerId = subject.getRepairedLayerId();

		/* Prepare the experts. */
		int[] expertIDs;
		if (useF1Selection) {
			expertIDs = f1SelectedExperts;
		} else {
			expertIDs = new int[MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS];
			for (int i = 0; i < MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS; i++) {
				expertIDs[i] = i;
			}
		}

		System.out.println("PATH:" + subject.getProjectPath());

		MNIST0_InternalData data = new MNIST0_InternalData(subject.getProjectPath(), "weights0.txt", "weights2.txt",
				"weights6.txt", "weights8.txt", "biases0.txt", "biases2.txt", "biases6.txt", "biases8.txt");
		Object repaired_weight_deltas = Z3SolutionParsing.loadRepairedWeights_MNIST0(subject.getRepairPath(),
				subject.getSolutionFileNamePrefix(), repairedLayerId, expertIDs,
				MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS);
		MNIST0_DNNt_Combined model = new MNIST0_DNNt_Combined(data, repaired_weight_deltas);
		MNIST0_DNNt_Original origModel = new MNIST0_DNNt_Original(data);

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
		File file = new File(subject.getLabelFilePath());
		BufferedReader br = new BufferedReader(new FileReader(file));
		String st;
		Integer[] labels = new Integer[60000];
		int index = 0;
		while ((st = br.readLine()) != null) {
			labels[index] = Integer.valueOf(st);
			index++;
			if (stopAfter != null && index == stopAfter) {
				break;
			}
		}
		br.close();

		/* Read input files and execute model. */
		file = new File(subject.getInputFilePath());
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
							if (subject.needsNormalization()) {
								input[i][j][k] = (double) (val / 255.0);
							} else {
								input[i][j][k] = (double) val;
							}
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
			Map<Integer, double[]> result = model.run(input, repairedLayerId, expertIDs, true);
			long timeCombinationNetwork = System.currentTimeMillis() - startTimeCombinationNetwork;
			accumulatedTimeCombinedNetwork += timeCombinationNetwork;

			// Combine NAIVE.
			long startTimeCombinationNAIVE = System.currentTimeMillis();
			int origLabelNAIVE = ExpertCombination.selectLabelWithMaxConfidence(result.get(-1));
			List<Integer> expertClaimsNAIVE = ExpertCombination.collectExpertClaims(expertIDs, result);
			ExpertCombination.combineExpertsByNaive(expertClaimsNAIVE, origLabelNAIVE);
			long timeNAIVE = System.currentTimeMillis() - startTimeCombinationNAIVE;
			accumulatedTimeNAIVE += timeNAIVE;
			accumulatedTimeNAIVETotal += (timeNAIVE + timeCombinationNetwork);
			System.out.print("; NAIVE=" + timeNAIVE + "; NAIVETotal=" + (timeNAIVE + timeCombinationNetwork));

			// Combine PREC.
			long startTimeCombinationPREC = System.currentTimeMillis();
			int origLabelPREC = ExpertCombination.selectLabelWithMaxConfidence(result.get(-1));
			List<Integer> expertClaimsPREC = ExpertCombination.collectExpertClaims(expertIDs, result);
			ExpertCombination.combineExpertsByPrecision(expertClaimsPREC, origLabelPREC, subject.getTrainPrecision());
			long timePREC = System.currentTimeMillis() - startTimeCombinationPREC;
			accumulatedTimePREC += timePREC;
			accumulatedTimePRECTotal += (timePREC + timeCombinationNetwork);
			System.out.print("; PREC=" + timePREC + "; PRECTotal=" + (timePREC + timeCombinationNetwork));

			// Combine VOTES.
			long startTimeCombinationVOTES = System.currentTimeMillis();
			int origLabelVOTES = ExpertCombination.selectLabelWithMaxConfidence(result.get(-1));
			List<Integer> expertClaimsVOTES = ExpertCombination.collectExpertClaims(expertIDs, result);
			ExpertCombination.combineExpertsByVotes(result, expertClaimsVOTES, origLabelVOTES, expertIDs,
					MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS);
			long timeVOTES = System.currentTimeMillis() - startTimeCombinationVOTES;
			accumulatedTimeVOTES += timeVOTES;
			accumulatedTimeVOTESTotal += (timeVOTES + timeCombinationNetwork);
			System.out.print("; VOTES=" + timeVOTES + "; VOTESTotal=" + (timeVOTES + timeCombinationNetwork));

			// Combine CONF.
			long startTimeCombinationCONF = System.currentTimeMillis();
			int origLabelCONF = ExpertCombination.selectLabelWithMaxConfidence(result.get(-1));
			List<Integer> expertClaimsCONF = ExpertCombination.collectExpertClaims(expertIDs, result);
			ExpertCombination.combineExpertsByConfidence(result, expertClaimsCONF, origLabelCONF);
			long timeCONF = System.currentTimeMillis() - startTimeCombinationCONF;
			accumulatedTimeCONF += timeCONF;
			accumulatedTimeCONFTotal += (timeCONF + timeCombinationNetwork);
			System.out.print("; CONF=" + timeCONF + "; CONFTotal=" + (timeCONF + timeCombinationNetwork));

			// Combine PVC.
			long startTimeCombinationPVC = System.currentTimeMillis();
			int origLabelPVC = ExpertCombination.selectLabelWithMaxConfidence(result.get(-1));
			List<Integer> expertClaimsPVC = ExpertCombination.collectExpertClaims(expertIDs, result);
			ExpertCombination.combineExpertsByPVC(result, expertClaimsPVC, origLabelPVC, subject.getTrainPrecision(),
					expertIDs, MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS);
			long timePVC = System.currentTimeMillis() - startTimeCombinationPVC;
			accumulatedTimePVC += timePVC;
			accumulatedTimePVCTotal += (timePVC + timeCombinationNetwork);
			System.out.print("; PVC=" + timePVC + "; PVCTotal=" + (timePVC + timeCombinationNetwork));

			System.out.println();
			count++;

			if (stopAfter != null && count == stopAfter) {
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

	public static void runOriginalCIFARDNN(SUBJECT subject, Integer stopAfter)
			throws NumberFormatException, IOException {

		System.out.println("PATH:" + subject.getProjectPath());

		CIFAR10_InternalData data = new CIFAR10_InternalData(subject.getProjectPath(), "weights0.txt", "weights2.txt",
				"weights5.txt", "weights7.txt", "weights11.txt", "weights13.txt", "biases0.txt", "biases2.txt",
				"biases5.txt", "biases7.txt", "biases11.txt", "biases13.txt");
		CIFAR10_DNNt_Original model = new CIFAR10_DNNt_Original(data);

		/* Initialize analytics */
		int passCounter = 0;
		int failCounter = 0;

		/* Read correct labels. */
		File file = new File(subject.getLabelFilePath());
		BufferedReader br = new BufferedReader(new FileReader(file));
		String st;
		Integer[] labels = new Integer[60000];
		int index = 0;
		while ((st = br.readLine()) != null) {
			labels[index] = Integer.valueOf(st);
			index++;
			if (stopAfter != null && index == stopAfter) {
				break;
			}
		}
		br.close();

		/* Read input files and execute model. */
		file = new File(subject.getInputFilePath());
		br = new BufferedReader(new FileReader(file));
		int count = 0;

		while ((st = br.readLine()) != null) {
			// System.out.println("INPUT:" + st);

			String[] values = st.split(",");
			double[][][] input = new double[32][32][3];
			index = 0;
			while (index < values.length) {
				for (int i = 0; i < 32; i++)
					for (int j = 0; j < 32; j++)
						for (int k = 0; k < 3; k++) {
							Double val = Double.valueOf(values[index]);
							index++;
							if (subject.needsNormalization()) {
								input[i][j][k] = (double) (val / 255.0);
							} else {
								input[i][j][k] = (double) val;
							}

						}
			}

			int origLabel = model.run(input);
			int correctLabel = labels[count];
			boolean passed = (origLabel == correctLabel);

			// Print results and collect analytics.
			System.out.println(count + "; IDEAL: " + correctLabel + "; ORIG: " + (passed ? "PASS" : "FAIL") + " "
					+ origLabel + "");
			if (passed) {
				passCounter++;
			} else {
				failCounter++;
			}

			count++;

			if (stopAfter != null && count == stopAfter) {
				break;
			}

		}

		br.close();

		// Calculate and print accuracy.
		System.out.println();
		System.out.println("COMBINATION;ACCURACY;PASS;FAIL");

		double accuracy = round((((double) passCounter) / (passCounter + failCounter)) * 100.0, 2);
		System.out.println("ORIG" + ";" + accuracy + ";" + passCounter + ";" + failCounter);
		System.out.println();

	}

	public static void runCIFAR10Experiment(SUBJECT subject, ExpertCombination.COMBINATION_METHOD combMethod)
			throws NumberFormatException, IOException {
		runCIFAR10Experiment(subject, combMethod, null, false);
	}

	public static void runCIFAR10Experiment(SUBJECT subject, ExpertCombination.COMBINATION_METHOD combMethod,
			Integer stopAfter, boolean useF1Selection) throws NumberFormatException, IOException {

		int repairedLayerId = subject.getRepairedLayerId(); // {0 | 2 | 6 | 8}

		/* Prepare the experts. */
		int[] expertIDs;
		if (useF1Selection) {
			expertIDs = subject.getF1SelectedExperts();
		} else {
			expertIDs = new int[CIFAR10_DNNt_Combined.NUMBER_OF_EXPERTS];
			for (int i = 0; i < CIFAR10_DNNt_Combined.NUMBER_OF_EXPERTS; i++) {
				expertIDs[i] = i;
			}
		}

		System.out.println("PATH:" + subject.getProjectPath());

		CIFAR10_InternalData data = new CIFAR10_InternalData(subject.getProjectPath(), "weights0.txt", "weights2.txt",
				"weights5.txt", "weights7.txt", "weights11.txt", "weights13.txt", "biases0.txt", "biases2.txt",
				"biases5.txt", "biases7.txt", "biases11.txt", "biases13.txt");
		Object repaired_weight_deltas = Z3SolutionParsing.loadRepairedWeights_CIFAR10(subject.getRepairPath(),
				repairedLayerId, expertIDs, CIFAR10_DNNt_Combined.NUMBER_OF_EXPERTS);
		CIFAR10_DNNt_Combined model = new CIFAR10_DNNt_Combined(data, repaired_weight_deltas);

		/* Initialize analytics */
		Map<Object, Integer> passCounter = new HashMap<>();
		Map<Object, Integer> failCounter = new HashMap<>();
		Map<Object, Integer> targetedPassCounter = new HashMap<>();
		Map<Object, Integer> targetedFailCounter = new HashMap<>();
		Map<Object, Integer> TPCounter = new HashMap<>();
		Map<Object, Integer> TNCounter = new HashMap<>();
		Map<Object, Integer> FPCounter = new HashMap<>();
		Map<Object, Integer> FNCounter = new HashMap<>();
		for (ExpertCombination.COMBINATION_METHOD x : ExpertCombination.COMBINATION_METHOD.values()) {
			passCounter.put(x, 0);
			failCounter.put(x, 0);
		}
		for (int x = 0; x < CIFAR10_DNNt_Combined.NUMBER_OF_EXPERTS + 2; x++) {
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
		File file = new File(subject.getLabelFilePath());
		BufferedReader br = new BufferedReader(new FileReader(file));
		String st;
		Integer[] labels = new Integer[60000];
		int index = 0;
		while ((st = br.readLine()) != null) {
			labels[index] = Integer.valueOf(st);
			index++;
			if (stopAfter != null && index == stopAfter) {
				break;
			}
		}
		br.close();

		/* Read input files and execute model. */
		file = new File(subject.getInputFilePath());
		br = new BufferedReader(new FileReader(file));
		int count = 0;

		while ((st = br.readLine()) != null) {
			// System.out.println("INPUT:" + st);

			String[] values = st.split(",");
			double[][][] input = new double[32][32][3];
			index = 0;
			while (index < values.length) {
				for (int i = 0; i < 32; i++)
					for (int j = 0; j < 32; j++)
						for (int k = 0; k < 3; k++) {
							Double val = Double.valueOf(values[index]);
							index++;
							if (subject.needsNormalization()) {
								input[i][j][k] = (double) (val / 255.0);
							} else {
								input[i][j][k] = (double) val;
							}

						}
			}

			Map<Integer, double[]> result = model.run(input, repairedLayerId, expertIDs);

			int correctLabel = labels[count];

			// Extract original decision.
			int origLabel = ExpertCombination.selectLabelWithMaxConfidence(result.get(-1)); /* ORIG */

			// Determine final decisions by experts.
			Map<ExpertCombination.COMBINATION_METHOD, Integer> results = ExpertCombination.combineExperts(combMethod,
					result, origLabel, subject.getTrainPrecision(), expertIDs, false,
					CIFAR10_DNNt_Combined.NUMBER_OF_EXPERTS);

			// Print results and collect analytics.
			System.out.print(count + "; IDEAL: " + correctLabel + "; ");
			for (Entry<ExpertCombination.COMBINATION_METHOD, Integer> combinedResult : results.entrySet()) {
				ExpertCombination.COMBINATION_METHOD currentCombinationMethod = combinedResult.getKey();
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
			for (int expertId = 0; expertId < CIFAR10_DNNt_Combined.NUMBER_OF_EXPERTS; expertId++) {
				int label = ExpertCombination.selectLabelWithMaxConfidence(result.get(expertId));
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

			if (stopAfter != null && count == stopAfter) {
				break;
			}

		}

		br.close();

		// Calculate and print accuracy.
		System.out.println();
		System.out.println("COMBINATION;ACCURACY;PASS;FAIL;TAR-ACC;TAR-PASS;TAR-FAIL;TP;TN;FP;FN;PREC;RECALL;F1");
		if (combMethod.equals(ExpertCombination.COMBINATION_METHOD.ALL)) {
			for (ExpertCombination.COMBINATION_METHOD combinationMethod : ExpertCombination.COMBINATION_METHOD
					.values()) {
				if (combinationMethod.equals(ExpertCombination.COMBINATION_METHOD.ALL)) {
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
		double[] prec = new double[CIFAR10_DNNt_Combined.NUMBER_OF_EXPERTS];
		for (int expertId = 0; expertId < CIFAR10_DNNt_Combined.NUMBER_OF_EXPERTS; expertId++) {
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

	public static void main(String[] args) {
		try {
			long startTime = System.currentTimeMillis();
//			runMNIST0Experiment(SUBJECT.LOW_QUALITY_PATTERN_TRAINING, ExpertCombination.COMBINATION_METHOD.ALL);
			runMNIST0Experiment(SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpD_TEST,
					ExpertCombination.COMBINATION_METHOD.ALL, 60000, false);
//			runMNIST0CombinationOverheadExperiment(SUBJECT.LOW_QUALITY_LAST_LAYER_TEST, 60000, false,
//					new int[] { 6, 8, 9 });
//			runOriginalCIFARDNN(SUBJECT.CIFAR_LAST_LAYER_ORIGINAL_TRAINING, 60000);
//			runOriginalCIFARDNN(SUBJECT.CIFAR_LAST_LAYER_TRAINING, 60000);
//			runCIFAR10Experiment(SUBJECT.CIFAR_LAST_LAYER_TEST, ExpertCombination.COMBINATION_METHOD.ALL, 10000, false);
			long totalRuntime = System.currentTimeMillis() - startTime;
			System.out.println();
			System.out.println("Total Runtime: " + totalRuntime + " ms");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
