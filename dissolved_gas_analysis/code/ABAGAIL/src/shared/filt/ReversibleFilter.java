package shared.filt;

import shared.DataSet;
import util.linalg.Matrix;
import util.linalg.Vector;

/**
 * A reversible filter is a filter that can be undone
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public interface ReversibleFilter extends DataSetFilter {
    /**
     * Perform the reverse on the given data set
     * @param set the set to reverse
     */
    public void reverse(DataSet set);
    
    public Matrix getProjection();
    
    public Vector getMean();
    

}
