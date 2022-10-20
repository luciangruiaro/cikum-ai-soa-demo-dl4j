package com.dl4jdemo.utils;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
public class ImgFace {
    private static final Logger log = LoggerFactory.getLogger(ImgFace.class);
    private INDArray imageFace;
    private BoundBox boundBox;
    private INDArray featureVector;

    public ImgFace(INDArray imageFace, BoundBox boundBox) {
        this.imageFace = imageFace;
        this.boundBox = boundBox;
    }

    public INDArray get() {
        return imageFace;
    }

}
