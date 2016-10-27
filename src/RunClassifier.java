
/**
 * @author Omer Faruk Arar
 * @date 05 June 2016
 */

import java.util.ArrayList;

public class RunClassifier {

	static ModifiedNaiveBayes mnb = new ModifiedNaiveBayes();

	public static void main(String[] args) {

		// @data satiri verilerin hemen ustunde yer almali, bu satira kadar reader ilerletilir
		// feature sayisi otomatik olarak hesaplanmaktadir.
		// ranges dosyasina gore aralik sayisi otomatik hesaplanmaktadir.
		String dataPath = "dataset/CM1_mdp.arff";

		int runTime = 20;
		int foldNumber = 10;
		double oversampleSize = 1;
		double normalizeMin = 0;				// normalize data with min value
		double normalizeMax = 10;				// normalize data with max value
		mnb.useFeaturesDependence = true;
		mnb.useFeaturesIndependence = false;

		mnb.loadData(dataPath);
		mnb.shuffleData();

		Results[] allResults = new Results[runTime*foldNumber];;
		Results result;
		int indx = 0;
		for (int j = 1; j <= runTime;j++ )
		{
			mnb.shuffleData();
			int totalTP = 0;
			int totalFP = 0;
			int totalTN = 0;
			int totalFN = 0;
			for (int i = 0; i < foldNumber;i++ )
			{
				mnb.splitTrainingTestSet(i,foldNumber);
				mnb.selectFeatures();
				mnb.normalizeTraininData(normalizeMin, normalizeMax);
				///bee.logNormalizeTraininData();
				mnb.discritize();
				mnb.oversampleTraingData(oversampleSize);
				mnb.createProbabilities();
				mnb.createDependentProbabilities();

				mnb.performance();
				totalTP += mnb.TP;
				totalFP += mnb.FP;
				totalTN += mnb.TN;
				totalFN += mnb.FN;

				double acc = (mnb.TP + mnb.TN) / (mnb.TP+mnb.FP+mnb.TN+mnb.FN);
				double recallCoeff = 1;
				double precision = mnb.TP / (double)(mnb.TP+mnb.FP); 
				double recall = mnb.TP / (double)(mnb.TP+mnb.FN); 
				double pf = mnb.FP / (double)(mnb.FP+mnb.TN); 
				double f1Score = ( (1+Math.pow(recallCoeff,2)) * recall * precision ) / (Math.pow(recallCoeff,2) * recall + precision);
				double bal = 1 - Math.sqrt((Math.pow((1-recall),2) + Math.pow((0-pf),2)) / 2);

				result = new Results((double)mnb.TP, (double)mnb.FP, (double)mnb.TN, (double)mnb.FN, precision, recall, pf, f1Score, bal);
				allResults[indx] = result;
				indx++;

				//	System.out.println("TP = " + bee.TP);
				//	System.out.println("FP = " + bee.FP);
				//	System.out.println("TN = " + bee.TN);
				//	System.out.println("FN = " + bee.FN);
			}

			System.out.println(runTime - j + "..");
			// System.out.println("TP = " + totalTP/foldNumber);
			// System.out.println("FP = " + totalFP/foldNumber);
			// System.out.println("TN = " + totalTN/foldNumber);
			// System.out.println("FN = " + totalFN/foldNumber);

			double acc = (totalTP + totalTN) / (totalTP+totalFP+totalTN+totalFN);
			double recallCoeff = 1;
			double precision = totalTP / (double)(totalTP+totalFP); 
			double recall = totalTP / (double)(totalTP+totalFN); 
			double pf = totalFP / (double)(totalFP+totalTN); 
			double f1Score = ( (1+Math.pow(recallCoeff,2)) * recall * precision ) / (Math.pow(recallCoeff,2) * recall + precision);
			double bal = 1 - Math.sqrt((Math.pow((1-recall),2) + Math.pow((0-pf),2)) / 2);

			// System.out.println("acc = " + acc);
			// System.out.println("precision = " + precision);
			// System.out.println("recall = " + recall);
			// System.out.println("pf = " + pf);
			// System.out.println("F = " + f1Score);
			// System.out.println("bal = " + bal);

		}

		calculateAndPrintResults(allResults, runTime*foldNumber);

	}

	static void calculateAndPrintResults(Results[] allResults, int n) {
		double totalTP = 0;
		double totalFP = 0;
		double totalTN = 0;
		double totalFN = 0;
		for(int i = 0; i < n; ++i) {
			totalTP += allResults[i].TP;
			totalFP += allResults[i].FP;
			totalTN += allResults[i].TN;
			totalFN += allResults[i].FN;
		}

		System.out.println("FINAL RESULT:");
		System.out.println("TP = " + totalTP/n);
		System.out.println("FP = " + totalFP/n);
		System.out.println("TN = " + totalTN/n);
		System.out.println("FN = " + totalFN/n);

		double acc = (totalTP + totalTN) / (totalTP+totalFP+totalTN+totalFN);
		double recallCoeff = 1;
		double precision = totalTP / (double)(totalTP+totalFP); 
		double recall = totalTP / (double)(totalTP+totalFN); 
		double pf = totalFP / (double)(totalFP+totalTN); 
		double f1Score = ( (1+Math.pow(recallCoeff,2)) * recall * precision ) / (Math.pow(recallCoeff,2) * recall + precision);
		double bal = 1 - Math.sqrt((Math.pow((1-recall),2) + Math.pow((0-pf),2)) / 2);

		System.out.println("acc = " + acc);
		System.out.println("precision = " + precision);
		System.out.println("recall = " + recall);
		System.out.println("pf = " + pf);
		System.out.println("F = " + f1Score);

		// calculate std dev of balance
		double sum = 0;
		double sq_sum = 0;
		for(int i = 0; i < n; ++i) {
			if (allResults[i].balance >= 0){
				sum += allResults[i].balance;
				sq_sum += allResults[i].balance * allResults[i].balance;
			}
		}
		double mean = sum / n;
		double variance = sq_sum / n - mean * mean;
		double stdDev = Math.sqrt(variance);
		System.out.println("bal = " + bal + "  STD DEV = " + stdDev);

	}
}
