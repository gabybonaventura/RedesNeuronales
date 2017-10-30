package ar.edu.unlam.eia.ml.diabetes;

/**
 * Created by TigerShark on 8/3/2017.
 */
public class TreeExperiments {

    public static void main(String[] args) throws Exception {

        ar.edu.unlam.eia.ml.diabetes.Classifier cl = new ar.edu.unlam.eia.ml.diabetes.Classifier("D:\\nn.model");

        System.out.println(cl.ClassifyInstance(new double[]{8,125,96,0,0,0.0,0.232,54}));

    }
}
