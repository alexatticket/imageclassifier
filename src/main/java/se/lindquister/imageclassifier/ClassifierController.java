package se.lindquister.imageclassifier;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.net.MalformedURLException;

@RestController
public class ClassifierController {

	@Autowired
	private TensorflowImageClassifierService tensorflowImageClassifierService;
	@Autowired
	private BoofCVImageClassifierService boofCVImageClassifierService;
	@Value("${useTensorflow}")
	private boolean useTensorflow;


	@RequestMapping("/classify")
	public ImageClassification classify(@RequestParam String url) throws MalformedURLException {
		ImageClassification classification;
		if (useTensorflow) {
			classification = tensorflowImageClassifierService.classify(url);
		} else {
			classification = boofCVImageClassifierService.classify(url);
		}
		return classification;
	}
}
