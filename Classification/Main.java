

public class Main {
	public static void main(String[] args) {
		try {
            //  first get the full features file
	    String[][] csvData = MyWekaUtils.readCSV("features_12f.csv");      
	    	//Select the features
	    	int[] selFea=MyWekaUtils.SeqFeatureSel(12,csvData,3);//3 for SVM
	    	System.out.println(selFea);
	    	//Print the features selected
	    	 for (int i = 0; i < selFea.length; i++) {	 
	 		    System.out.println(i);
	    		 System.out.println(selFea[i]);}

			int[] features = selFea; 
			//Converting .csv file to .arff file for Weka classifier
            String arffData = MyWekaUtils.csvToArff(csvData, features);
            //Classify
            double accuracy = MyWekaUtils.classify(arffData, 3);
            //Calculate accuracy
            System.out.println(accuracy);
                        
        } catch (Exception ex) {
            System.out.println(ex.toString());
        }
	}
}
