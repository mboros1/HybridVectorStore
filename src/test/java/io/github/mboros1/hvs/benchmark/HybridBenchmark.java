package io.github.mboros1.hvs.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.mboros1.hvs.index.HybridIndexer;
import io.github.mboros1.hvs.search.HybridSearcher;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Ad-hoc benchmark that indexes the entire {@code samples/} corpus and then executes a batch of
 * hybrid queries to get a coarse view of throughput.
 *
 * <p>Usage (from repository root):
 *
 * <pre>{@code mvn -DskipTests package exec:java \
 *   -Dexec.mainClass=io.github.mboros1.hvs.benchmark.HybridBenchmark \
 *   -Dexec.classpathScope=test}</pre>
 *
 * <p>The benchmark is intentionally simple and aims to catch high-level regressions, not to produce
 * publishable performance figures.
 */
public final class HybridBenchmark {

  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final String DOC_SUFFIX = ".doc.ndjson";
  private static final String VEC_SUFFIX = ".vec.ndjson";
  private static final int TOKENS_PER_DOC = 12;
  private static final int VECTOR_SAMPLE_LIMIT = 4096;
  private static final int DEFAULT_QUERY_COUNT = 10_000;
  private static final int DEFAULT_TOP_K = 10;

  private HybridBenchmark() {}

  public static void main(String[] args) throws Exception {
    BenchmarkConfig config = BenchmarkConfig.parse(args);
    runBenchmark(config);
  }

  private static void runBenchmark(BenchmarkConfig config) throws Exception {
    Path samplesDir = Paths.get("samples");
    if (!Files.isDirectory(samplesDir)) {
      throw new IllegalStateException("samples/ directory not found. Run benchmark at repo root.");
    }

    System.out.printf(Locale.ROOT, "Benchmark config: %s%n", config);

    Path workRoot = Files.createTempDirectory("hybrid-bench-");
    Path indexDir = workRoot.resolve("index");
    Path outputDir = workRoot.resolve("out");
    Path sidecarDir = outputDir.resolve("sidecars");
    Files.createDirectories(indexDir);
    Files.createDirectories(outputDir);
    Files.createDirectories(sidecarDir);

    HybridIndexer indexer = new HybridIndexer();
    int maxVectorDim = indexer.maxVectorDimension();

    List<HybridIndexer.ShardInput> shards = new ArrayList<>();
    List<String> textTokens = new ArrayList<>();
    CorpusStats corpusStats =
        collectShardsAndTokens(samplesDir, sidecarDir, shards, textTokens, maxVectorDim);

    System.out.printf(
        Locale.ROOT,
        "Prepared %d shard pairs (%d docs, %.2f million tokens candidate pool)%n",
        shards.size(),
        corpusStats.docCount,
        textTokens.size() / 1_000_000.0);

    long indexStart = System.nanoTime();
    HybridIndexer.IndexStats stats =
        indexer.index(new HybridIndexer.IndexRequest(shards, indexDir, outputDir));
    long indexEnd = System.nanoTime();

    double indexSeconds = Duration.ofNanos(indexEnd - indexStart).toMillis() / 1000.0;
    double docsPerSecond =
        stats.docsIndexed() > 0 ? stats.docsIndexed() / indexSeconds : Double.NaN;
    System.out.printf(
        Locale.ROOT,
        "Indexing complete: docs=%d vectors=%d truncated=%s duplicates=%d time=%.2fs (%.2f docs/s)%n",
        stats.docsIndexed(),
        stats.vectorsIndexed(),
        stats.truncated(),
        stats.duplicateVectors(),
        indexSeconds,
        docsPerSecond);
    System.out.printf(
        Locale.ROOT,
        "Index manifest: %s%nPaths registry: %s%n",
        stats.manifestPath(),
        outputDir.resolve("paths.json"));

    List<float[]> vectorSamples =
        collectVectorSamples(shards, maxVectorDim, VECTOR_SAMPLE_LIMIT, config.randomSeed);
    System.out.printf(
        Locale.ROOT,
        "Collected %d vector samples for query workload (max dimension %d)%n",
        vectorSamples.size(),
        maxVectorDim);

    List<QueryTask> queryTasks =
        buildQueryTasks(
            textTokens,
            vectorSamples,
            (int) stats.docsIndexed(),
            config.totalQueries,
            config.randomSeed);
    if (queryTasks.isEmpty()) {
      System.out.println("No queries generated; skipping search benchmark.");
      return;
    }

    try (HybridSearcher searcher =
        new HybridSearcher(indexDir, outputDir.resolve("paths.json"))) {
      runQueryBenchmark(searcher, queryTasks, config);
    }

    System.out.printf(Locale.ROOT, "Benchmark working directory: %s%n", workRoot);
  }

  private static CorpusStats collectShardsAndTokens(
      Path samplesDir,
      Path sidecarDir,
      Collection<HybridIndexer.ShardInput> shards,
      List<String> tokenSink,
      int maxVectorDim)
      throws IOException {
    long docCount = 0;
    long totalTokens = 0;
    try (DirectoryStream<Path> docStream =
        Files.newDirectoryStream(samplesDir, "*" + DOC_SUFFIX)) {
      for (Path docPath : docStream) {
        String fileName = docPath.getFileName().toString();
        String base = fileName.substring(0, fileName.length() - DOC_SUFFIX.length());
        Path vecPath = docPath.getParent().resolve(base + VEC_SUFFIX);
        if (!Files.exists(vecPath)) {
          continue;
        }
        Path sidecarPath = sidecarDir.resolve(base + ".idx");
        Files.createDirectories(sidecarPath.getParent());
        shards.add(HybridIndexer.ShardInput.of(docPath, vecPath).withSidecar(sidecarPath));

        try (BufferedReader reader = Files.newBufferedReader(docPath)) {
          String line;
          while ((line = reader.readLine()) != null) {
            JsonNode node = MAPPER.readTree(line);
            String text = extractText(node);
            if (!text.isBlank()) {
              totalTokens += extractTokens(text, tokenSink, TOKENS_PER_DOC);
            }
            docCount++;
          }
        }
      }
    }
    return new CorpusStats(docCount, totalTokens, maxVectorDim);
  }

  private static List<float[]> collectVectorSamples(
      Collection<HybridIndexer.ShardInput> shards,
      int maxDimension,
      int limit,
      long randomSeed)
      throws IOException {
    List<float[]> samples = new ArrayList<>(Math.min(limit, 1024));
    Random random = new Random(randomSeed ^ 0x5DEECE66DL);
    long seen = 0;
    for (HybridIndexer.ShardInput shard : shards) {
      Path vecPath = shard.vectorPath();
      try (BufferedReader reader = Files.newBufferedReader(vecPath)) {
        String line;
        while ((line = reader.readLine()) != null) {
          JsonNode node = MAPPER.readTree(line);
          JsonNode arr = node.path("embeddings");
          if (!arr.isArray() || arr.size() == 0) {
            continue;
          }
          float[] vector = truncate(arr, maxDimension);
          if (samples.size() < limit) {
            samples.add(vector);
          } else {
            long position = seen++;
            int r = random.nextInt((int) (position + 1));
            if (r < limit) {
              samples.set(r, vector);
            }
          }
        }
      }
    }
    return samples;
  }

  private static List<QueryTask> buildQueryTasks(
      List<String> tokens,
      List<float[]> vectors,
      int fallbackCount,
      int totalQueries,
      long randomSeed) {
    List<QueryTask> tasks = new ArrayList<>(totalQueries);
    if (tokens.isEmpty() && vectors.isEmpty()) {
      return tasks;
    }
    Random random = new Random(randomSeed);
    for (int i = 0; i < totalQueries; i++) {
      String text =
          tokens.isEmpty() ? null : tokens.get(random.nextInt(tokens.size()));
      float[] vector =
          vectors.isEmpty() ? null : vectors.get(random.nextInt(vectors.size()));
      if (text == null && vector == null && !tokens.isEmpty()) {
        text = tokens.get(i % tokens.size());
      }
      tasks.add(new QueryTask(text, vector));
    }
    return tasks;
  }

  private static void runQueryBenchmark(
      HybridSearcher searcher, List<QueryTask> tasks, BenchmarkConfig config) throws IOException {
    int warmup = Math.min(config.warmupQueries, tasks.size());
    if (warmup > 0) {
      for (int i = 0; i < warmup; i++) {
        executeQuery(searcher, tasks.get(i), config.topK);
      }
    }

    long start = System.nanoTime();
    for (QueryTask task : tasks) {
      executeQuery(searcher, task, config.topK);
    }
    long end = System.nanoTime();
    double elapsedSeconds = Duration.ofNanos(end - start).toMillis() / 1000.0;
    double avgMillis = (end - start) / 1_000_000.0 / tasks.size();
    double qps = tasks.size() / elapsedSeconds;

    System.out.printf(
        Locale.ROOT,
        "Query benchmark: queries=%d topK=%d elapsed=%.2fs avg=%.3fms throughput=%.2f qps%n",
        tasks.size(),
        config.topK,
        elapsedSeconds,
        avgMillis,
        qps);
  }

  private static void executeQuery(HybridSearcher searcher, QueryTask task, int topK)
      throws IOException {
    HybridSearcher.SearchRequest.Builder builder =
        HybridSearcher.SearchRequest.builder()
            .topK(topK)
            .bm25TopK(Math.max(topK * 5, 100))
            .annTopK(Math.max(topK * 5, 100))
            .fetchFullDocument(false);
    if (task.text() != null) {
      builder.text(task.text());
    }
    if (task.vector() != null) {
      builder.vector(task.vector());
    }
    searcher.search(builder.build());
  }

  private static String extractText(JsonNode node) {
    String text = nullableText(node.get("text"));
    if (text == null || text.isBlank()) {
      text = nullableText(node.get("content"));
    }
    return text != null ? text : "";
  }

  private static String nullableText(JsonNode node) {
    return node != null && !node.isNull() ? node.asText() : null;
  }

  private static int extractTokens(String text, List<String> sink, int limit) {
    if (text.isBlank()) {
      return 0;
    }
    int added = 0;
    String[] parts =
        text.replaceAll("[^\\p{IsAlphabetic}\\p{IsDigit}]+", " ")
            .trim()
            .split("\\s+");
    for (String token : parts) {
      if (token.length() < 3) {
        continue;
      }
      sink.add(token.toLowerCase(Locale.ROOT));
      added++;
      if (added >= limit) {
        break;
      }
    }
    return added;
  }

  private static float[] truncate(JsonNode arrayNode, int maxDimension) {
    int len = Math.min(arrayNode.size(), maxDimension);
    float[] vector = new float[len];
    double norm = 0d;
    for (int i = 0; i < arrayNode.size(); i++) {
      double v = arrayNode.get(i).asDouble();
      if (i < len) {
        vector[i] = (float) v;
        norm += v * v;
      }
    }
    if (norm > 0d) {
      float scale = (float) (1.0d / Math.sqrt(norm));
      for (int i = 0; i < len; i++) {
        vector[i] *= scale;
      }
    }
    return vector;
  }

  private record CorpusStats(long docCount, long tokensCollected, int maxVectorDim) {}

  private record QueryTask(String text, float[] vector) {}

  private record BenchmarkConfig(
      int totalQueries, int warmupQueries, int topK, long randomSeed) {

    static BenchmarkConfig parse(String[] args) {
      int totalQueries = DEFAULT_QUERY_COUNT;
      int warmupQueries = Math.min(500, DEFAULT_QUERY_COUNT / 10);
      int topK = DEFAULT_TOP_K;
      long seed = 0xC0FFEE;
      for (String arg : args) {
        if (arg.startsWith("--queries=")) {
          totalQueries = Integer.parseInt(arg.substring("--queries=".length()));
        } else if (arg.startsWith("--warmup=")) {
          warmupQueries = Integer.parseInt(arg.substring("--warmup=".length()));
        } else if (arg.startsWith("--topK=")) {
          topK = Integer.parseInt(arg.substring("--topK=".length()));
        } else if (arg.startsWith("--seed=")) {
          seed = Long.parseLong(arg.substring("--seed=".length()));
        }
      }
      return new BenchmarkConfig(totalQueries, warmupQueries, topK, seed);
    }

    @Override
    public String toString() {
      return String.format(
          Locale.ROOT,
          "queries=%d warmup=%d topK=%d seed=%d",
          totalQueries,
          warmupQueries,
          topK,
          randomSeed);
    }
  }
}
