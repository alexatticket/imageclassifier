package se.lindquister.imageclassifier;

import java.util.List;

public class ImageClassification {
	private List<CategoryAndScore> categoryAndScores;

	public ImageClassification(List<CategoryAndScore> categoryAndScores) {
		this.categoryAndScores = categoryAndScores;
	}

	public List<CategoryAndScore> getCategoryAndScores() {
		return categoryAndScores;
	}
}
