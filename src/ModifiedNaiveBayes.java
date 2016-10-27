
/**
 * @author Omer Faruk Arar
 * @date 05 June 2016
 */

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericTransform;

public class ModifiedNaiveBayes {

	Instances allData;
	Instances trainingData;
	Instances testData;
	Discretize discFilter;
	private int instanceNumberOfEachFold; 
	private ArrayList rangesData = new ArrayList(); 
	int featureNumber;
	int TP, TN, FP, FN; // True Positive, True Negative, ...
	boolean useFeaturesDependence;
	boolean useFeaturesIndependence;

	int yesCounter, noCounter;
	ArrayList<MinMax[]> featureDiscrites =  new ArrayList<MinMax[]>(); 

	double yesProbabilities[][];
	double yesCounterList[][];
	double noCounterList[][];
	int numberOfBins;
	private ArrayList binNumbers = new ArrayList();
	double[][] featureMeanValuesYes;
	double[][] featureVarianceValuesYes;
	double[][] featureStdevValuesYes;
	double[][] featureMeanValuesNo;
	double[][] featureVarianceValuesNo;
	double[][] featureStdevValuesNo;

	private double[][][][] dependentNoCounterList;
	private double[][][][] dependentYesCounterList;
	ArrayList<String> predictedResults = new ArrayList<String>();

	void loadData(String path){

		// read arff file

		try {
			BufferedReader reader = new BufferedReader( new FileReader(path));
			allData = new Instances(reader);
			reader.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// setting class attribute
		allData.setClassIndex(allData.numAttributes() - 1);
		featureNumber = allData.numAttributes()-1;

	}

	public void shuffleData() {
		Collections.shuffle(allData);
	}

	public void createDiscritizationArray()
	{
		binNumbers.clear();
		featureDiscrites.clear();
		numberOfBins = 0;

		for(int i = 0; i < featureNumber; i++)
		{
			if(discFilter.getCutPoints(i) == null)
			{
				binNumbers.add(1);
				continue;
			}
			else{
				binNumbers.add(discFilter.getCutPoints(i).length+1);

				if( numberOfBins < (discFilter.getCutPoints(i).length+1) )
					numberOfBins = discFilter.getCutPoints(i).length + 1;
			}

		}


		MinMax featureRanges[] = new MinMax[numberOfBins];
		String[] ins;
		double minVal=0, maxVal=0;

		for(int i = 0; i < featureNumber; i++)
		{
			featureRanges = new MinMax[numberOfBins];

			int jIndex = 0;
			boolean first = true;
			for(int j = 0; j < (int)binNumbers.get(i)-1 ; j++)
			{
				if(first)
				{
					minVal = (trainingData.attributeStats(0).numericStats.min)-1;
					first = false;
					j--;
				}
				else
				{
					maxVal = discFilter.getCutPoints(i)[j];
					featureRanges[j] = new MinMax(minVal, maxVal);
					minVal = maxVal;
					jIndex = j+1;
				}

			}

			if((int)binNumbers.get(i) == 1)
				minVal = (trainingData.attributeStats(0).numericStats.min)-1;
			else
				minVal = discFilter.getCutPoints(i)[jIndex-1];

			maxVal = (trainingData.attributeStats(0).numericStats.max)+1;
			featureRanges[jIndex] = new MinMax(minVal, maxVal);


			featureDiscrites.add(featureRanges);
		}
	}

	public void loadRangesData(String rangesPath) {

		try {
			BufferedReader br = new BufferedReader(new FileReader(rangesPath));
			String line;

			while ((line = br.readLine()) != null) {
				String[] parts = line.split(",");
				rangesData.add(parts);
				binNumbers.add(parts.length/2);

				if(numberOfBins < (parts.length / 2)){
					numberOfBins = parts.length / 2;
				}
			}

			br.close();


		} catch ( FileNotFoundException e ) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		catch ( IOException e ) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		MinMax featureRanges[] = new MinMax[numberOfBins];
		String[] ins;
		double minVal, maxVal;

		for(int i = 0; i < rangesData.size(); i++)
		{
			featureRanges = new MinMax[numberOfBins];
			ins = (String[]) rangesData.get(i);

			for(int j = 0; j < ins.length; j++)
			{
				minVal = Double.parseDouble(ins[j]);
				j++;
				maxVal = Double.parseDouble(ins[j]);
				featureRanges[j/2] = new MinMax(minVal, maxVal);
			}

			featureDiscrites.add(featureRanges);
		}

	}

	void splitTrainingTestSet(int foldId, int foldNumber) {

		trainingData = allData.trainCV(foldNumber, foldId);
		testData = allData.testCV(foldNumber, foldId);

		yesCounter = (trainingData.attributeStats(trainingData.classIndex()).nominalCounts)[0];
		noCounter = (trainingData.attributeStats(trainingData.classIndex()).nominalCounts)[1];

		Collections.shuffle(trainingData);

		//getMean();
		//getVariance();
		//getStdDev();

	}

	public void createProbabilities() {

		initCounterList();

		double val;
		double minVal, maxVal;
		double realResult;
		int numberOfBins1;
		for (int i = 0; i < trainingData.size(); i++)
		{
			realResult = trainingData.instance(i).classValue();
			for(int j = 0; j < featureNumber; j++)
			{
				val = trainingData.get(i).value(j);
				numberOfBins1 = (Integer) binNumbers.get(j);
				for(int k = 0; k < numberOfBins1; k++)
				{
					minVal = featureDiscrites.get(j)[k].getMin();
					maxVal = featureDiscrites.get(j)[k].getMax();

					if(val >= minVal && val < maxVal)
					{
						if(realResult == 0.0)
							yesCounterList[j][k]++;
						else
							noCounterList[j][k]++;

						break;
					}

				}
			}

		}

	}

	private void initCounterList() {
		yesProbabilities = new double[featureNumber][numberOfBins];
		yesCounterList = new double[featureNumber][numberOfBins];
		noCounterList = new double[featureNumber][numberOfBins];

		for(int i = 0; i < featureNumber; i++)
		{
			for(int j = 0; j < numberOfBins; j++)
			{
				yesCounterList[i][j] = 1;
				noCounterList[i][j] = 1;

			}
		}

	}

	private void initDependentCounterList() {
		dependentYesCounterList = new double[featureNumber][featureNumber][numberOfBins][numberOfBins];
		dependentNoCounterList = new double[featureNumber][featureNumber][numberOfBins][numberOfBins];

		for(int i = 0; i < featureNumber; i++)
		{
			for(int j = 0; j < numberOfBins; j++)
			{
				for(int k = 0; k < featureNumber; k++)
				{
					for(int m = 0; m < numberOfBins; m++)
					{
						dependentYesCounterList[i][k][j][m] = 1;
						dependentNoCounterList[i][k][j][m] = 1;
					}
				}

			}
		}

	}

	public void createDependentProbabilities() {

		initDependentCounterList();

		double realResult;
		double val;
		double minVal, maxVal;
		String[] ins;
		int numberOfBins1, numberOfBins2;

		for (int i = 0; i < trainingData.size(); i++)
		{
			realResult = trainingData.instance(i).classValue();
			for(int j = 0; j < featureNumber; j++)
			{
				val = trainingData.get(i).value(j);

				numberOfBins1 = (Integer) binNumbers.get(j);
				for(int k = 0; k < numberOfBins1; k++)
				{
					minVal = featureDiscrites.get(j)[k].getMin();
					maxVal = featureDiscrites.get(j)[k].getMax();

					if(val >= minVal && val < maxVal)
					{
						for(int t = 0; t < featureNumber; t++)
						{
							val = trainingData.get(i).value(t);
							numberOfBins2 = (Integer) binNumbers.get(t);
							for(int s = 0; s < numberOfBins2; s++)
							{
								minVal = featureDiscrites.get(t)[s].getMin();
								maxVal = featureDiscrites.get(t)[s].getMax();
								if(val >= minVal && val < maxVal)
								{
									if(realResult == 0.0)
										dependentYesCounterList[j][t][k][s]++;
									else
										dependentNoCounterList[j][t][k][s]++;
								}
							}
						}
					}

				}
			}


		}


	}

	public void performance() {

		TP=TN=FP=FN=0;
		double realResult;
		double val;
		double minVal, maxVal;
		double yesProbabilityVal, noProbabilityVal;
		double dependentYesProbabilityVal, dependentNoProbabilityVal;
		int yesCnt, noCnt;
		double likelihoodOfYes, likelihoodOfNo;
		int numberOfBins1, numberOfBins2;

		for (int i = 0; i < testData.size(); i++)
		{
			yesCnt=0; noCnt=0;
			realResult = testData.instance(i).classValue();
			yesProbabilityVal = noProbabilityVal = 1;
			dependentYesProbabilityVal = dependentNoProbabilityVal = 1;
			likelihoodOfYes = likelihoodOfNo = 1;

			for(int j = 0; j < featureNumber; j++)
			{
				val = testData.get(i).value(j);
				numberOfBins1 = (Integer) binNumbers.get(j);
				for(int k = 0; k < numberOfBins1; k++)
				{
					minVal = featureDiscrites.get(j)[k].getMin();
					maxVal = featureDiscrites.get(j)[k].getMax();

					if(val >= minVal && val < maxVal)
					{
						if(yesCounterList[j][k] + noCounterList[j][k] != 0)
						{
							yesProbabilityVal *= yesCounterList[j][k] / (yesCounterList[j][k] + noCounterList[j][k]);
							noProbabilityVal *= noCounterList[j][k] / (yesCounterList[j][k] + noCounterList[j][k]);

						}

						//likelihoodOfYes *= naiveFunctionYes(val, j, k) * (double) yesCounterList[j][k] / (yesCounterList[j][k] + noCounterList[j][k]);
						//likelihoodOfNo *= naiveFunctionNo(val, j, k) * (double) noCounterList[j][k] / (yesCounterList[j][k] + noCounterList[j][k]); 

						for(int t = 0; t < featureNumber; t++)
						{
							if(t != j)
							{
								val = testData.get(i).value(t);
								numberOfBins2 = (Integer) binNumbers.get(t);
								for(int s = 0; s < numberOfBins2; s++)
								{
									minVal = featureDiscrites.get(t)[s].getMin();
									maxVal = featureDiscrites.get(t)[s].getMax();
									if(val >= minVal && val < maxVal)
									{

										if(dependentYesCounterList[j][t][k][s] + dependentNoCounterList[j][t][k][s] != 0)
										{
											if(dependentYesCounterList[j][t][k][s] != 0)
											{
												dependentYesProbabilityVal *= dependentYesCounterList[j][t][k][s] / (dependentYesCounterList[j][t][k][s] + dependentNoCounterList[j][t][k][s]);
												yesCnt += dependentYesCounterList[j][t][k][s];
											}
											if(dependentNoCounterList[j][t][k][s] != 0){
												dependentNoProbabilityVal *= dependentNoCounterList[j][t][k][s] / (dependentYesCounterList[j][t][k][s] + dependentNoCounterList[j][t][k][s]);
												noCnt += dependentNoCounterList[j][t][k][s];
											}

										}
									}
								}
							}


						}
					}

				}

			}


			double yesCoeff = 1;
			double noCoeff = 1;

			if(useFeaturesIndependence)
			{
				yesCoeff = yesProbabilityVal / noProbabilityVal;
				noCoeff = noProbabilityVal / yesProbabilityVal;

				if(yesProbabilityVal != 0 && noProbabilityVal == 0)
				{
					yesCoeff = yesProbabilityVal;
					noCoeff = 1;
				}
				else if(yesProbabilityVal == 0 && noProbabilityVal != 0)
				{
					yesCoeff = 1;
					noCoeff = noProbabilityVal;
				}
				else if(yesProbabilityVal == 0 && noProbabilityVal == 0)
				{
					yesCoeff = 1;
					noCoeff = 1;
				}
			}

			// dependent calculation
			double dependentYesCoeff = 1;
			double dependentNoCoeff = 1;

			if(useFeaturesDependence)
			{
				dependentYesCoeff = dependentYesProbabilityVal / dependentNoProbabilityVal;
				dependentNoCoeff = dependentNoProbabilityVal / dependentYesProbabilityVal;

				if(dependentYesProbabilityVal != 0 && dependentNoProbabilityVal == 0)
				{
					dependentYesCoeff = dependentYesProbabilityVal;
					dependentNoCoeff = 1;
				}
				else if(dependentYesProbabilityVal == 0 && dependentNoProbabilityVal != 0)
				{
					dependentYesCoeff = 1;
					dependentNoCoeff = dependentNoProbabilityVal;
				}
				else if(dependentYesProbabilityVal == 0 && dependentNoProbabilityVal == 0)
				{
					dependentYesCoeff = 1;
					dependentNoCoeff = 1;
				}

			}


			double totalYesCoeff = yesCoeff * dependentYesCoeff;
			double totalNoCoeff = noCoeff * dependentNoCoeff;

			yesProbabilityVal = totalYesCoeff;
			noProbabilityVal = totalNoCoeff;

			yesProbabilityVal *= (double) yesCounter / (yesCounter + noCounter);
			noProbabilityVal *= (double) noCounter / (yesCounter + noCounter);


			// performance
			if(realResult == 0.0 && yesProbabilityVal >= noProbabilityVal )
			{
				predictedResults.add("TP");
				TP++;
			}
			else if(realResult == 0.0 && yesProbabilityVal < noProbabilityVal )
			{
				predictedResults.add("FN");
				FN++;
			}
			else if(realResult == 1.0 && yesProbabilityVal >= noProbabilityVal)
			{
				predictedResults.add("FP");
				FP++;
			}
			else if(realResult == 1.0 && yesProbabilityVal < noProbabilityVal)
			{
				predictedResults.add("TN");
				TN++;
			}
			else{
				System.out.println("dependentYes:" + dependentYesProbabilityVal);
				System.out.println("dependentNo:" + dependentNoProbabilityVal);
				System.out.println("Yes:" + yesProbabilityVal);
				System.out.println("No:" + noProbabilityVal);
			}

		}


	}

	private double naiveFunctionNo(double x, int fIndex, int binIndex) {
		double result;
		double e, phi2, z, s;
		double mean = featureMeanValuesNo[fIndex][binIndex];
		double stdev = featureStdevValuesNo[fIndex][binIndex];
		double sqrt_pi2 = 2.506628274631; /* sqrt(2*pi) */

		if (Double.isNaN(x)) {
			return Double.NaN;
		}
		if (Double.isNaN(mean) || Double.isNaN(stdev)) {
			return Double.NaN;
		} else {
			z = x;
			z -= mean;
			s = stdev;
			phi2 = sqrt_pi2 * s;
			e = -0.5 * (z * z) / (s * s);
			result = Math.exp(e) / phi2;
			if(Double.isNaN(result))
				result = 1;
			return result;
		}
	}

	private double naiveFunctionYes(double x, int fIndex, int binIndex) {
		double result;
		double e, phi2, z, s;
		double mean = featureMeanValuesYes[fIndex][binIndex];
		double stdev = featureStdevValuesYes[fIndex][binIndex];
		double sqrt_pi2 = 2.506628274631; /* sqrt(2*pi) */

		if (Double.isNaN(x)) {
			return Double.NaN;
		}
		if (Double.isNaN(mean) || Double.isNaN(stdev)) {
			return Double.NaN;
		} else {
			z = x;
			z -= mean;
			s = stdev;
			phi2 = sqrt_pi2 * s;
			e = -0.5 * (z * z) / (s * s);
			result = Math.exp(e) / phi2;
			if(Double.isNaN(result))
				result = 1;
			return result;
		}
	}

	public void normalizeTraininData(double normalizeMin, double normalizeMax) {

		double translation = normalizeMin;
		double scale = normalizeMax - normalizeMin;
		Normalize norm = new Normalize();
		norm.setTranslation(translation);
		norm.setScale(scale);
		try {
			norm.setInputFormat(trainingData);
			trainingData = Filter.useFilter(trainingData, norm);
			testData = Filter.useFilter(testData, norm);
		} catch (Exception e) {

			e.printStackTrace();
		}

	}

	public void logNormalizeTraininData() {


	}

	public void oversampleTraingData(double oversampleSize) {

		Resample sampleFilter = new Resample();
		sampleFilter.setBiasToUniformClass(1.0); 
		sampleFilter.setSampleSizePercent((double) (100*oversampleSize));
		try {
			sampleFilter.setInputFormat(trainingData);
			trainingData = Filter.useFilter(trainingData, sampleFilter); 
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}


		//		double result;  // 0.0 no (false); 1.0 yes (true)
		//		Instance newIns;
		//
		//		Instances sampledTrain = new Instances(trainingData);
		//
		//		for(int i = 0; i < trainingData.numInstances(); i++)
		//		{
		//			result = trainingData.instance(i).classValue();
		//
		//			if(result == 0.0)
		//			{
		//				for(int j = 0; j < oversampleSize; j++)
		//				{
		//					newIns = trainingData.instance(i);
		//					sampledTrain.add(newIns);
		//				}
		//			}
		//
		//		}
		//
		//		trainingData.clear();
		//		trainingData = sampledTrain;

		//smote
		//		SMOTE smote = new SMOTE();
		//		smote.setPercentage((double) (100*oversampleSize));
		//		try {
		//			smote.setInputFormat(trainingData);
		//			trainingData = Filter.useFilter(trainingData, smote); 
		//		} catch (Exception e) {
		//			// TODO Auto-generated catch block
		//			e.printStackTrace();
		//		}

	}

	public void selectFeatures() {
		AttributeSelection fsFilter = new AttributeSelection();  // package weka.filters.supervised.attribute!
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		fsFilter.setEvaluator(eval); 
		fsFilter.setSearch(search);
		try {
			fsFilter.setInputFormat(trainingData);
			trainingData = Filter.useFilter(trainingData, fsFilter);
			testData = Filter.useFilter(testData, fsFilter);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		//pca unsupervised
		//		PrincipalComponents pca = new PrincipalComponents();
		//		try {
		//			pca.setInputFormat(trainingData);
		//			trainingData = Filter.useFilter(trainingData, pca);
		//			testData = Filter.useFilter(testData, pca);
		//		} catch (Exception e) {
		//			// TODO Auto-generated catch block
		//			e.printStackTrace();
		//		}

		trainingData.setClassIndex(trainingData.numAttributes() - 1);
		featureNumber = trainingData.numAttributes() - 1;


	}

	public void discritize() {
		Instances discTrain;
		discFilter = new Discretize();
		try {
			discFilter.setUseKononenko(true);
			discFilter.setInputFormat(trainingData);
			discTrain = Filter.useFilter(trainingData, discFilter);
			//discTest = Filter.useFilter(normTest, discFilter);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		createDiscritizationArray();

	}


}
