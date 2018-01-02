package se.lindquister.imageclassifier;

import java.net.MalformedURLException;

public interface ImageClassifierService {
	ImageClassification classify(String url) throws MalformedURLException;
}
