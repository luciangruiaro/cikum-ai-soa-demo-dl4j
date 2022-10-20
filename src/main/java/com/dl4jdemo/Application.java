package com.dl4jdemo;

import com.dl4jdemo.utils.ImgFace;
import com.dl4jdemo.featurebank.FeatureBank;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.event.ApplicationStartedEvent;
import org.springframework.context.event.EventListener;
import org.springframework.core.io.FileUrlResource;
import org.springframework.core.io.Resource;

import java.io.File;
import java.io.IOException;
import java.util.Optional;
import java.util.Scanner;

@SpringBootApplication
public class Application {
    private static final Logger logger = LoggerFactory.getLogger(Application.class);
    private Resource[] trainImages;
    private FeatureBank featureBank;
    private FaceRecognizer faceRecognizer;

    @Autowired
    public Application(
            @Value("classpath:images/dataset/train/*/*.*") Resource[] trainImgs,
            @Qualifier(FeatureBank.DATA_SET) FeatureBank featureKnowledge,
            FaceRecognizer faceCatcher
    ) {
        this.trainImages = trainImgs;
        this.featureBank = featureKnowledge;
        this.faceRecognizer = faceCatcher;
    }

    public static void main(String[] args) {
        SpringApplication
                .run(Application.class, args)
                .close();
    }

    @EventListener
    private void onApplicationStartup(ApplicationStartedEvent event) throws IOException {
        logger.info("Building feature vectors --- start");
        for (Resource trainImage : trainImages) {
            var faceFeatures = faceRecognizer.getFaceFeatures(trainImage);
            for (ImgFace imgFace : faceFeatures.getImgFaces()) {
                var label = getLabel(trainImage);
                featureBank.put(label, imgFace.getFeatureVector());
            }
        }
        logger.info("Building feature vectors --- DONE");

        Scanner sc = new Scanner(System.in);
        while(true) {
            logger.info("Input image path (type exit to close):");
            String inputLine = sc.nextLine();
            if (inputLine.equalsIgnoreCase("exit")) {
                break;
            }
            Resource resource = new FileUrlResource(inputLine);
            var faceFeatures = faceRecognizer.getFaceFeatures(resource);
            for (ImgFace imgFace : faceFeatures.getImgFaces()) {
                featureBank.getSimilar(imgFace.getFeatureVector());
            }
        }
    }

    private String getLabel(Resource resource) throws IOException {
        return Optional
                .ofNullable(resource)
                .map(Resource::getFilename)
                .map(resource.getURL().getPath()::split)
                .filter(urlParts ->
                        urlParts.length != 0)
                .map(urlParts ->
                        urlParts[0].split(File.separator))
                .filter(urlParts ->
                        urlParts.length != 0)
                .map(urlParts ->
                        urlParts[urlParts.length - 1])
                .orElseThrow();
    }

}
