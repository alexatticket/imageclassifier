package se.lindquister.imageclassifier;

import boofcv.abst.scene.ImageClassifier;
import boofcv.factory.scene.ClassifierAndSource;
import boofcv.factory.scene.FactoryImageClassifier;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.io.image.UtilImageIO;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.Planar;
import deepboof.io.DeepBoofDataBaseOps;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import javax.annotation.PostConstruct;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

@RestController
public class ClassifierController {
	private ImageClassifier<Planar<GrayF32>> classifier;

	@PostConstruct
	public void init() throws IOException {
		ClassifierAndSource cs = FactoryImageClassifier.nin_imagenet(); // Test set 62.6% for 1000 categories

		File path = DeepBoofDataBaseOps.downloadModel(cs.getSource(), new File("download_data"));

		classifier = cs.getClassifier();
		classifier.loadModel(path);
	}


	@RequestMapping("/classify")
	public ImageClassification classify(@RequestParam String url) throws MalformedURLException {

		BufferedImage buffered = UtilImageIO.loadImage(new URL(url));
		if (buffered == null)
			throw new RuntimeException("Couldn't find input image");

		Planar<GrayF32> image = new Planar<>(GrayF32.class, buffered.getWidth(), buffered.getHeight(), 3);
		ConvertBufferedImage.convertFromPlanar(buffered, image, true, GrayF32.class);

		classifier.classify(image);
		System.out.println(classifier.getCategories().get(classifier.getBestResult()));


		return getImageClassification(classifier);
	}

	private ImageClassification getImageClassification(ImageClassifier<Planar<GrayF32>> classifier) {
		List<CategoryAndScore> categoryAndScores = new ArrayList<>();
		for (int i = 0; i < 5; i++) {
			ImageClassifier.Score score = classifier.getAllResults().get(i);
			categoryAndScores.add(new CategoryAndScore(score.score, classifier.getCategories().get(score.category)));
		}
		return new ImageClassification(categoryAndScores);
	}
}
