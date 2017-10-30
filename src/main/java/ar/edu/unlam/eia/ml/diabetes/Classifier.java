package ar.edu.unlam.eia.ml.diabetes;

import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Created by TigerShark on 8/7/2017.
 */
public class Classifier {

    private FilteredClassifier fc;
    private Instances testData;

    public Classifier(String modelFilePath) throws Exception {

        this.fc = (FilteredClassifier) weka.core.SerializationHelper.read(modelFilePath);

        ArrayList<Attribute> attributes = new ArrayList<>(9);
        attributes.add(new Attribute("Number of times pregnant"));
        attributes.add(new Attribute("Plasma glucose concentration"));
        attributes.add(new Attribute("Diastolic blood pressure"));
        attributes.add(new Attribute("Triceps skin fold thickness"));
        attributes.add(new Attribute("2-Hour serum insulin"));
        attributes.add(new Attribute("Body mass index"));
        attributes.add(new Attribute("Diabetes pedigree function"));
        attributes.add(new Attribute("Age"));

        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("positive");
        classValues.add("negative");
        attributes.add(new Attribute("Class", classValues));

        this.testData = new Instances("Test-Set-A", attributes, 1);

        this.testData.setClassIndex(testData.numAttributes() - 1);

    }

    public String ClassifyInstance(double[] attributeValues) throws Exception {
        this.testData.clear();

        Instance instance = new DenseInstance(9);
        for (int i = 0; i < attributeValues.length; i++) {
            instance.setValue(i, attributeValues[i]);
        }

        instance.setDataset(this.testData);
        this.testData.add(instance);

        return this.testData.classAttribute().value((int) this.fc.classifyInstance(instance));
    }
}