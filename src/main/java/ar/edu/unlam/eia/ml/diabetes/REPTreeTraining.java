package ar.edu.unlam.eia.ml.diabetes;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Created by TigerShark on 8/3/2017.
 */
public class REPTreeTraining {

    public static void main(String[] args) throws Exception {

        // Levantamos el dataset "diabetes.csv"
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("datasets/diabetes.csv");

        // Obtenemos las instancias (juegos de datos)
        Instances train = source.getDataSet();

        // Seteamos cual atributo (columna) es la "clase" (si tiene o no diabetes)
        train.setClassIndex(train.numAttributes() - 1);

        // Obtenemos una copia de las instancias para testear el entrenamiento
        Instances test = source.getDataSet();
        test.setClassIndex(train.numAttributes() - 1);
        //////////////////////////////////////////////////////////////////////
        Random rnd = new Random();
        int valor1;
    	int valor2;
    	int valor3;
        
        do {
        	valor1 = rnd.nextInt(8);
        	valor2 = rnd.nextInt(8);
        	valor3 = rnd.nextInt(8);
        }while (valor1 == valor2 || valor1 == valor3 || valor2 == valor3);
        
        // Creamos filtros para no usar algunos de los atributos (columnas)
        System.out.println(valor1 + " " +valor2 + " " +valor3);
        Remove rm = new Remove();
        rm.setAttributeIndicesArray(new int[]{valor1, valor2, valor3}); // Removemos columnas agregando su indice aqui
        rm.setInputFormat(train);
        ///////////////////////////////////////////////////////////////////


        // Creamos un clasificador usando el algoritmo de arbol RepTree
        Classifier tree = new weka.classifiers.trees.REPTree();

        // Creamos un meta-clasificador que aplica los filtros antes de usar la data
        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(rm);
        fc.setClassifier(tree);
        ////////////////////////////////////////////////////////////////////////////

        // Entrenamos el arbol (ajustamos el "modelo")
        fc.buildClassifier(train);

        // Serializamos el modelo para usar en el tp de agentes
        String location = "C:\\Users\\Gabriel\\Desktop\\TP IA - Redes neuranales\\ParteB_Punto3";
        weka.core.SerializationHelper.write(location + "\\Diabetes" +valor1 + ""  + valor2 + "" + valor3, fc);
        ///////////////////////////////////////////////////////
        float VerdaderoNegativo = 0, VerdaderoPositivo=0, FalsoNegativo=0, FalsoPositivo =0;
        
        // Imprimimos los resultados sobre la misma data (para calcular error de entrenamiento)
        for (int i = 0; i < test.numInstances(); i++) {
            double pred = fc.classifyInstance(test.instance(i));
            System.out.print("ID: " + i);
            
            String actual = test.classAttribute().value((int) test.instance(i).classValue());
            String predicho = test.classAttribute().value((int) pred);
            
            if (actual.equals(predicho)) {
            	if(predicho.equals("positive")) {
            		VerdaderoPositivo++;
            	}else VerdaderoNegativo++;
            }else if(predicho.equals("negative")) {
            	FalsoNegativo++;
            }else 
            	FalsoPositivo++;
            
            System.out.print(", actual: " + actual);
            System.out.println(", predicted: " + predicho);
        }

        System.out.println("VerdaderoPositivo: " +VerdaderoPositivo + " VerdaderoNegativo: " +VerdaderoNegativo + " FalsoNegativo: " + FalsoNegativo + " FalsoPositivo: " + FalsoPositivo);
        float precision = VerdaderoPositivo/(VerdaderoPositivo+FalsoPositivo);
        float recall = VerdaderoPositivo/(VerdaderoPositivo+FalsoNegativo);
        float F1Score = 2*((precision*recall)/(precision+recall));
        System.out.println("Precision: " + precision + "\nRecall: " + recall + "\nF1Score: " + F1Score);
    }
}
