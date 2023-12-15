

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.meta.AdaBoostM1;

import weka.core.Instances;
//J48 [4, 6, 3, 8, 9] [69.33333333333333, 75.66666666666667, 79.5, 84.66666666666667, 85.5]
// Random Forest [4, 6, 2, 3, 1, 8] [66.33333333333333, 75.66666666666667, 84.0, 88.66666666666667, 90.16666666666667, 90.83333333333333]
// SVM [] [4, 3, 8, 0] [71.83333333333333, 76.33333333333333, 79.66666666666667, 80.5]
//// 1 83.83% 2 90% 3 81%
//int[] features = {0, 1, 2, 3, 4, 5,6,7,8,9,10,11}; // all the features
/**
 *
 * @author mm5gg
 */
public class MyWekaUtils {

    public static double classify(String arffData, int option) throws Exception {
		StringReader strReader = new StringReader(arffData);
		Instances instances = new Instances(strReader);
		strReader.close();
		instances.setClassIndex(instances.numAttributes() - 1);
		
		Classifier classifier;
		if(option==1)
			classifier = new J48(); // Decision Tree classifier
		else if(option==2)			
			classifier = new RandomForest();
		else if(option == 3)
			classifier = new SMO();  //This is a SVM classifier
		else if(option == 4)
			classifier = new AdaBoostM1();
		else
			return -1;
		
		classifier.buildClassifier(instances); // build classifier
		
		Evaluation eval = new Evaluation(instances);
		eval.crossValidateModel(classifier, instances, 10, new Random(1), new Object[] { });
		
		return eval.pctCorrect();
	}
    
    
    public static String[][] readCSV(String filePath) throws Exception {
        StringBuilder sb = new StringBuilder();
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        ArrayList<String> lines = new ArrayList();
        String line;

        while ((line = br.readLine()) != null) {
            lines.add(line);;
        }


        if (lines.size() == 0) {
            System.out.println("No data found");
            return null;
        }

        int lineCount = lines.size();

        String[][] csvData = new String[lineCount][];
        String[] vals;
        int i, j;
        for (i = 0; i < lineCount; i++) {            
                csvData[i] = lines.get(i).split(",");            
        }
        
        return csvData;

    }

    public static String csvToArff(String[][] csvData, int[] featureIndices) throws Exception {
        int total_rows = csvData.length;
        int total_cols = csvData[0].length;
        int fCount = featureIndices.length;
        String[] attributeList = new String[fCount + 1];
        int i, j;
        for (i = 0; i < fCount; i++) {
            attributeList[i] = csvData[0][featureIndices[i]];
        }
        attributeList[i] = csvData[0][total_cols - 1];

        String[] classList = new String[1];
        classList[0] = csvData[1][total_cols - 1];

        for (i = 1; i < total_rows; i++) {
            classList = addClass(classList, csvData[i][total_cols - 1]);
        }

        StringBuilder sb = getArffHeader(attributeList, classList);

        for (i = 1; i < total_rows; i++) {
            for (j = 0; j < fCount; j++) {
                sb.append(csvData[i][featureIndices[j]]);
                sb.append(",");
            }            
            sb.append(csvData[i][total_cols - 1]);
            sb.append("\n");
        }

        return sb.toString();
    }

    private static StringBuilder getArffHeader(String[] attributeList, String[] classList) {
        StringBuilder s = new StringBuilder();
        s.append("@RELATION wada\n\n");

        int i;
        for (i = 0; i < attributeList.length - 1; i++) {
            s.append("@ATTRIBUTE ");
            s.append(attributeList[i]);
            s.append(" numeric\n");
        }

        s.append("@ATTRIBUTE ");
        s.append(attributeList[i]);
        s.append(" {");
        s.append(classList[0]);

        for (i = 1; i < classList.length; i++) {
            s.append(",");
            s.append(classList[i]);
        }
        s.append("}\n\n");
        s.append("@DATA\n");
        return s;
    }

    private static String[] addClass(String[] classList, String className) {
        int len = classList.length;
        int i;
        for (i = 0; i < len; i++) {
            if (className.equals(classList[i])) {
                return classList;
            }
        }

        String[] newList = new String[len + 1];
        for (i = 0; i < len; i++) {
            newList[i] = classList[i];
        }
        newList[i] = className;

        return newList;
    }
    
//    private static boolean isFeatureSelected(int feature, int[] featureList) {
//        for (int i : featureList) {
//            if (i == feature) {
//                return true;
//            }
//        }
//        return false;
//    }
//    
//    private static int[] addFeature(int feature, int[] featureList) {
//        int[] newList = new int[featureList.length];
//        System.arraycopy(featureList, 0, newList, 0, featureList.length);
//        newList[featureList.length] = feature;
//        return newList;
//    }
    
//    public static int[] SeqFeatureSel(int numFeatures, String[][] csvData, int classifierM) throws Exception {
//        int[] featureList = new int[numFeatures];
//        for (int i = 0; i < numFeatures; i++) {
//            featureList[i] = -1;
//        }
//       
//        double bestAcc = 0;
//        int bestFea = -1;
//
//        for (int j = 0; j < numFeatures; j++) {
//            String arffData = csvToArff(csvData, new int[] { j });
//            double acc = classify(arffData, classifierM);
//
//            if (acc > bestAcc) {
//                bestAcc = acc;
//                bestFea = j;
//            }
//        }
//
//        featureList[0] = bestFea;
//        
//        return featureList;
//    }

    
    @SuppressWarnings("rawtypes")
	public static int[] SeqFeatureSel(int numFeatures,String[][] csvData,int classifierM) throws Exception {
    	ArrayList<Integer> featureList = new ArrayList();
    	ArrayList<Double> allAcc = new ArrayList();

    	//int pre;
    	
    	for(int i=0;i<numFeatures;i++) {//AT MOST 12 turns
    		
    		double bestAcc=0;
    		int bestFea=0;
//    		System.out.println("i:");
//    		System.out.println(i);
//    		System.out.println();
    		for(int j=0;j< numFeatures; j++) {//Add
	    		if(!featureList.contains(j)) {//another one
	    	    	ArrayList<Integer> currentList = new ArrayList<>(featureList);
	    			currentList.add(j);
	    			int[] currentArr=currentList.stream().mapToInt(Integer::valueOf).toArray();
//	    			System.out.println("j:");
//	    			System.out.print(j);
//	    			System.out.println();
	    			
	    			String arffData = csvToArff(csvData,currentArr);
//	    			System.out.println(arffData);
	    			double acc=classify(arffData,classifierM);
//	    			System.out.println(currentArr.length);
//	    			System.out.println(acc);
//	    			if(j==11)
//	    				System.out.println(acc);
//	    			System.out.println();
//	    			for (int k = 0; k < currentArr.length; k++) {	 
//	    				System.out.println("Test:"); 
//	    				System.out.println(currentArr[k]);}
	    			if(acc>bestAcc) {
	    				bestAcc=acc;
	        			bestFea=j;	
	    			}
	
	    		}
    		}
//    		System.out.println("bestAcc");
//    		System.out.println(bestAcc);
    		if(allAcc.size()<1) {
    			allAcc.add(bestAcc);
    			featureList.add(bestFea);
    		}else{
    			if(bestAcc-allAcc.get(allAcc.size()-1)<1) {//bestAcc>=100
    				//System.out.println("check");
    				allAcc.add(bestAcc);
        			featureList.add(bestFea);
    				break;
 
    			}
    			allAcc.add(bestAcc);
    			featureList.add(bestFea);
    		}
    	}
    	int[] featureArray=featureList.stream().mapToInt(Integer::valueOf).toArray();
    	System.out.println(featureList);
    	System.out.println(allAcc);
		return featureArray;
    }
    
}
