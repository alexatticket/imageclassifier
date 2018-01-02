package se.lindquister.imageclassifier;


import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;
import org.tensorflow.*;
import org.tensorflow.types.UInt8;

import javax.annotation.PostConstruct;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;


@Service
public class TensorflowImageClassifierService implements ImageClassifierService {

	private Graph g;
	private List<String> labels;

	@Value("${tf.modelFileName}")
	private String modelName;
	@Value("${tf.labelsFileName}")
	private String labelName;
	@Value("${tfFileLocation}")
	private String tfFileLocation;

	@PostConstruct
	public void init() throws IOException {
		byte[] graphDef = readAllBytesOrExit(Paths.get(new ClassPathResource(tfFileLocation + "/" + modelName).getURI()));
		labels = readAllLinesOrExit(Paths.get(new ClassPathResource(tfFileLocation + "/" + labelName).getURI()));
		g = new Graph();
		g.importGraphDef(graphDef);
	}


	@Override
	public ImageClassification classify(String url) {

		byte[] imageBytes = new byte[0];
		try {
			imageBytes = getImageByteArrayFromUrl(url);
		} catch (IOException e) {
			e.printStackTrace();
		}

		Tensor<Float> image = constructAndExecuteGraphToNormalizeImage(imageBytes);
		float[] labelProbabilities = executeInceptionGraph(image);
		int bestLabelIdx = maxIndex(labelProbabilities);

		System.out.println(
				String.format("BEST MATCH: %s (%.2f%% likely)",
						labels.get(bestLabelIdx),
						labelProbabilities[bestLabelIdx] * 100f));


		return new ImageClassification(getTopMatches(labelProbabilities));
	}

	private List<CategoryAndScore> getTopMatches(float[] labelProbabilities) {
		List<CategoryAndScore> all = new ArrayList<>();
		for (int i = 0; i < labelProbabilities.length; i++) {
			all.add(new CategoryAndScore(labelProbabilities[i], labels.get(i)));
		}
		all.sort(Comparator.comparingDouble(CategoryAndScore::getScore).reversed());
		return all;
	}


	private byte[] getImageByteArrayFromUrl(String urlString) throws IOException {
		URL url = new URL(urlString);
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		InputStream is = null;
		try {
			is = url.openStream();
			byte[] byteChunk = new byte[4096]; // Or whatever size you want to read in at a time.
			int n;

			while ((n = is.read(byteChunk)) > 0) {
				baos.write(byteChunk, 0, n);
			}
		} catch (IOException e) {
			System.err.printf("Failed while reading bytes from %s: %s", url.toExternalForm(), e.getMessage());
			e.printStackTrace();
			// Perform any other exception handling that's appropriate.
		} finally {
			if (is != null) {
				is.close();
			}
		}
		return baos.toByteArray();
	}


	private Tensor<Float> constructAndExecuteGraphToNormalizeImage(byte[] imageBytes) {
		Graph g = new Graph();
		GraphBuilder b = new GraphBuilder(g);
		// Some constants specific to the pre-trained model at:
		// https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
		//
		// - The model was trained with images scaled to 224x224 pixels.
		// - The colors, represented as R, G, B in 1-byte each were converted to
		//   float using (value - Mean)/Scale.
		final int H = 224;
		final int W = 224;
		final float mean = 117f;
		final float scale = 1f;

		// Since the graph is being constructed once per execution here, we can use a constant for the
		// input image. If the graph were to be re-used for multiple input images, a placeholder would
		// have been more appropriate.
		final Output<String> input = b.constant("input", imageBytes);
		final Output<Float> output =
				b.div(
						b.sub(
								b.resizeBilinear(
										b.expandDims(
												b.cast(b.decodeJpeg(input, 3), Float.class),
												b.constant("make_batch", 0)),
										b.constant("size", new int[]{H, W})),
								b.constant("mean", mean)),
						b.constant("scale", scale));
		try (Session s = new Session(g)) {
			return s.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
		}

	}

	private float[] executeInceptionGraph(Tensor<Float> image) {
		Session s = new Session(g);
		Tensor<Float> result = s.runner()
				.feed("input", image)
				.fetch("final_result") // this is the name of the output operation for our model
				.run().get(0)
				.expect(Float.class);
		final long[] rshape = result.shape();
		if (result.numDimensions() != 2 || rshape[0] != 1) {
			throw new RuntimeException(
					String.format(
							"Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
							Arrays.toString(rshape)));
		}
		int nlabels = (int) rshape[1];
		return result.copyTo(new float[1][nlabels])[0];
	}


	private static int maxIndex(float[] probabilities) {
		int best = 0;
		for (int i = 1; i < probabilities.length; ++i) {
			if (probabilities[i] > probabilities[best]) {
				best = i;
			}
		}
		return best;
	}

	private static byte[] readAllBytesOrExit(Path path) {
		try {
			return Files.readAllBytes(path);
		} catch (IOException e) {
			System.err.println("Failed to read [" + path + "]: " + e.getMessage());
			System.exit(1);
		}
		return null;
	}

	private static List<String> readAllLinesOrExit(Path path) {
		try {
			return Files.readAllLines(path, Charset.forName("UTF-8"));
		} catch (IOException e) {
			System.err.println("Failed to read [" + path + "]: " + e.getMessage());
			System.exit(0);
		}
		return null;
	}

	// In the fullness of time, equivalents of the methods of this class should be auto-generated from
	// the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
	// like Python, C++ and Go.
	static class GraphBuilder {
		private Graph g;

		GraphBuilder(Graph g) {
			this.g = g;
		}

		Output<Float> div(Output<Float> x, Output<Float> y) {
			return binaryOp("Div", x, y);
		}

		<T> Output<T> sub(Output<T> x, Output<T> y) {
			return binaryOp("Sub", x, y);
		}

		<T> Output<Float> resizeBilinear(Output<T> images, Output<Integer> size) {
			return binaryOp3("ResizeBilinear", images, size);
		}

		<T> Output<T> expandDims(Output<T> input, Output<Integer> dim) {
			return binaryOp3("ExpandDims", input, dim);
		}

		<T, U> Output<U> cast(Output<T> value, Class<U> type) {
			DataType dtype = DataType.fromClass(type);
			return g.opBuilder("Cast", "Cast")
					.addInput(value)
					.setAttr("DstT", dtype)
					.build()
					.output(0);
		}

		Output<UInt8> decodeJpeg(Output<String> contents, long channels) {
			return g.opBuilder("DecodeJpeg", "DecodeJpeg")
					.addInput(contents)
					.setAttr("channels", channels)
					.build()
					.output(0);
		}

		<T> Output<T> constant(String name, Object value, Class<T> type) {
			try (Tensor<T> t = Tensor.<T>create(value, type)) {
				return g.opBuilder("Const", name)
						.setAttr("dtype", DataType.fromClass(type))
						.setAttr("value", t)
						.build()
						.output(0);
			}
		}

		Output<String> constant(String name, byte[] value) {
			return this.constant(name, value, String.class);
		}

		Output<Integer> constant(String name, int value) {
			return this.constant(name, value, Integer.class);
		}

		Output<Integer> constant(String name, int[] value) {
			return this.constant(name, value, Integer.class);
		}

		Output<Float> constant(String name, float value) {
			return this.constant(name, value, Float.class);
		}

		private <T> Output<T> binaryOp(String type, Output<T> in1, Output<T> in2) {
			return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T>output(0);
		}

		private <T, U, V> Output<T> binaryOp3(String type, Output<U> in1, Output<V> in2) {
			return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T>output(0);
		}
	}


}
