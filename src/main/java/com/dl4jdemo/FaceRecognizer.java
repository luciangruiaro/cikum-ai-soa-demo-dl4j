package com.dl4jdemo;

import com.dl4jdemo.models.Dl4jModel;
import com.dl4jdemo.models.mtcnn.Mtcnn;
import com.dl4jdemo.utils.FaceFeatures;
import com.dl4jdemo.utils.ImgFace;
import com.dl4jdemo.utils.ImageUtils;
import com.dl4jdemo.utils.Nd4jUtils;
import com.dl4jdemo.models.InceptionResNetV1;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;
import java.util.List;

import static java.util.stream.Collectors.toList;

/**
 * Getting Face Features
 * <p>
 * Process consist two stages:
 * 1. Detect faces on image using MTCNN
 * 2. Extract face features from detected face
 */

@Component
public class FaceRecognizer {
    private static final Logger logger = LoggerFactory.getLogger(FaceRecognizer.class);
    private NativeImageLoader loader = new NativeImageLoader();
    private Mtcnn mtcnn;
    private ImageUtils imageUtils;
    private Dl4jModel model;
    private ComputationGraph faceFeatureExtracter;
    private Nd4jUtils nd4jUtils;

    @Autowired
    public FaceRecognizer(Mtcnn mtcnn, InceptionResNetV1 model, ImageUtils imageUtils, Nd4jUtils nd4jUtils) {
        this.mtcnn = mtcnn;
        this.imageUtils = imageUtils;
        this.model = model;
        this.nd4jUtils = nd4jUtils;
    }

    @PostConstruct
    public void init() throws IOException {
        faceFeatureExtracter = model.getGraph();
    }

    /**
     * Detect faces on img
     *
     * @param img img file
     * @return array of detected images
     * @throws IOException exception while file is read
     */

    public List<ImgFace> recognizeFaces(Resource img) throws IOException {
        try (InputStream is = img.getInputStream()) {
            var imgMatrix = loader.asMatrix(is);
            return mtcnn
                    .detectFaces(imgMatrix)
                    .stream()
                    .map(boundBox -> {
                        var imageFace = nd4jUtils.crop(boundBox, imgMatrix);
                        return new ImgFace(imageFace, boundBox);
                    })
                    .collect(toList());
        }
    }

    /**
     * Extract features for each face in array
     *
     * @param faces INDArray represent faces on image (image size depend on model)
     * @return list of face feature vectors
     */
    public List<ImgFace> extractFeatures(List<ImgFace> faces) {
        logger.info("Extract features from faces : {}", faces.size());
        faces.stream().parallel().forEach(imgFace -> {
            var resizedFace = Nd4jUtils.imresample(imgFace.get(), model.inputHeight(), model.inputWidth());
            var output = faceFeatureExtracter.output(resizedFace)[1];
            imgFace.setFeatureVector(output);
        });
        return faces;
    }

    /**
     * Method combine detect faces and extract features for each face in array
     *
     * @param img img file
     * @return list of face feature vectors
     * @throws IOException exception while file is read
     */

    public FaceFeatures getFaceFeatures(Resource img) throws IOException {
        logger.info("start : {}", img.getFilename());
        var detectedFaces = recognizeFaces(img);

        if (detectedFaces == null || detectedFaces.isEmpty()) {
            logger.warn("There is no face recognized in image : {}", img);
            return new FaceFeatures(img, Collections.emptyList());
        }

        if (logger.isDebugEnabled()) {
            logger.debug("Write detected face image");
            for (int i = 0; i < detectedFaces.size(); i++) {
                imageUtils.toFile(detectedFaces.get(i).getImageFace(), "jpg", i + "_" + img.getFilename());
            }
        }
        var imageFaces = extractFeatures(detectedFaces);
        logger.info("end : {}", img.getFilename());
        return new FaceFeatures(img, imageFaces);
    }
}