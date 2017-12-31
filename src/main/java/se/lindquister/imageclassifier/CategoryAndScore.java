package se.lindquister.imageclassifier;

public class CategoryAndScore {
	private final String category;
	private final double score;

	public CategoryAndScore(double score, String category) {
		this.score = score;
		this.category = category;
	}

	public String getCategory() {
		return category;
	}

	public double getScore() {
		return score;
	}
}
