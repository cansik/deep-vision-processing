package ch.bildspur.vision.result;

import java.util.ArrayList;
import java.util.List;

public class ResultList<T> extends ArrayList<T> implements NetworkResult {

    public ResultList() {
        super();
    }

    public ResultList(List<T> list) {
        super();
        addAll(list);
    }
}
