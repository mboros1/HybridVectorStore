package io.github.mboros1.hvs.integration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import io.github.mboros1.hvs.index.HybridIndexer;
import io.github.mboros1.hvs.search.HybridSearcher;

final class HybridIndexerSearcherIntegrationTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final String DOC_SUFFIX = ".doc.ndjson";
  private static final String VEC_SUFFIX = ".vec.ndjson";

  @TempDir Path tempDir;

  @Test
  void indexAndSearchUsingRealSampleShard() throws IOException {
    Path samplesDir = Paths.get("samples");
    Assumptions.assumeTrue(
        Files.isDirectory(samplesDir), "samples/ directory must exist for integration test");

    HybridIndexer indexer = new HybridIndexer();
    int maxVectorDim = indexer.maxVectorDimension();
    SampleContext context = locateSampleWithVectors(samplesDir, maxVectorDim);
    Assumptions.assumeTrue(context != null, "No matching doc/vec shard pair found under samples/");

    SamplePair pair = context.pair();
    SampleFixture fixture = context.fixture();

    Path indexDir = tempDir.resolve("index");
    Path outputDir = tempDir.resolve("out");
    Path sidecar = tempDir.resolve(pair.baseName() + ".idx");

    HybridIndexer.ShardInput shardInput =
        HybridIndexer.ShardInput.of(pair.docPath(), pair.vecPath()).withSidecar(sidecar);
    HybridIndexer.IndexRequest request =
        new HybridIndexer.IndexRequest(List.of(shardInput), indexDir, outputDir);
    HybridIndexer.IndexStats stats = indexer.index(request);

    assertTrue(stats.docsIndexed() > 0, "Expected at least one document indexed");
    assertEquals(
        stats.docsIndexed(),
        stats.vectorsIndexed(),
        "Vector count should match indexed document count");
    assertEquals(
        fixture.projectedVector().length,
        stats.dimension(),
        "Vector dimension should match sample embedding length");

    Path manifest = outputDir.resolve("manifest.json");
    Path pathsJson = outputDir.resolve("paths.json");
    assertTrue(Files.exists(manifest), "manifest.json should be written");
    assertTrue(Files.exists(pathsJson), "paths.json should be written");

    try (HybridSearcher searcher = new HybridSearcher(indexDir, pathsJson)) {
      HybridSearcher.SearchRequest searchRequest =
          HybridSearcher.SearchRequest.builder()
              .text(fixture.queryToken())
              .vector(fixture.projectedVector())
              .fetchFullDocument(true)
              .topK(10)
              .bm25TopK(25)
              .annTopK(25)
              .build();

      List<HybridSearcher.SearchResult> results = searcher.search(searchRequest);
      assertFalse(results.isEmpty(), "Hybrid search should return hits for sample shard");

      HybridSearcher.SearchResult match =
          results.stream()
              .filter(r -> r.id().equals(fixture.docId()))
              .findFirst()
              .orElse(results.get(0));

      assertEquals(
          fixture.docId(),
          match.id(),
          "Expected to retrieve the shard document matching the sample vector");
      assertTrue(
          match.fullDocument().isPresent(), "Full document payload should be available on demand");

      String payload = match.fullDocument().orElseThrow();
      JsonNode parsed = MAPPER.readTree(payload);
      assertEquals(fixture.docId(), parsed.path("id").asText());
      String parsedText = parsed.path("text").asText(parsed.path("content").asText(""));
      assertTrue(
          parsedText.contains(fixture.queryToken()),
          "Retrieved document text should contain the selected query token");
    }
  }

  private static SampleContext locateSampleWithVectors(Path samplesDir, int maxDimension)
      throws IOException {
    try (DirectoryStream<Path> docStream = Files.newDirectoryStream(samplesDir, "*" + DOC_SUFFIX)) {
      for (Path docPath : docStream) {
        String name = docPath.getFileName().toString();
        if (!name.endsWith(DOC_SUFFIX)) {
          continue;
        }
        String base = name.substring(0, name.length() - DOC_SUFFIX.length());
        Path vecPath = docPath.getParent().resolve(base + VEC_SUFFIX);
        if (Files.exists(vecPath)) {
          SampleFixture fixture = tryReadFixture(docPath, vecPath, maxDimension);
          if (fixture != null) {
            return new SampleContext(new SamplePair(docPath, vecPath, base), fixture);
          }
        }
      }
    }
    return null;
  }

  private static SampleFixture tryReadFixture(Path docPath, Path vecPath, int maxDimension)
      throws IOException {
    String firstDocLine;
    try (BufferedReader docReader =
        Files.newBufferedReader(docPath, StandardCharsets.UTF_8)) {
      firstDocLine = docReader.readLine();
    }
    if (firstDocLine == null) {
      return null;
    }

    JsonNode docNode = MAPPER.readTree(firstDocLine);
    String docId = docNode.path("id").asText(null);
    if (docId == null || docId.isBlank()) {
      return null;
    }
    String text = docNode.path("text").asText(docNode.path("content").asText(""));
    String token = chooseQueryToken(text, docId);

    VectorInfo vectorInfo = findVector(vecPath, docId, maxDimension);
    if (vectorInfo == null || vectorInfo.projected().length == 0) {
      return null;
    }
    return new SampleFixture(docId, token, vectorInfo.originalDimension(), vectorInfo.projected());
  }

  private static VectorInfo findVector(Path vecPath, String docId, int maxDimension)
      throws IOException {
    try (BufferedReader vecReader =
        Files.newBufferedReader(vecPath, StandardCharsets.UTF_8)) {
      String line;
      while ((line = vecReader.readLine()) != null) {
        JsonNode vectorNode = MAPPER.readTree(line);
        if (!docId.equals(vectorNode.path("id").asText(null))) {
          continue;
        }
        JsonNode embeddings = vectorNode.path("embeddings");
        if (!embeddings.isArray() || embeddings.size() == 0) {
          return null;
        }
        ArrayNode array = (ArrayNode) embeddings;
        int originalDimension = array.size();
        int effective = Math.min(originalDimension, maxDimension);
        float[] vector = new float[effective];
        for (int i = 0; i < effective; i++) {
          vector[i] = array.get(i).floatValue();
        }
        return new VectorInfo(originalDimension, vector);
      }
    }
    return null;
  }

  private static String chooseQueryToken(String text, String fallback) {
    if (text == null) {
      return fallback;
    }
    String[] tokens = text.replaceAll("[^\\p{IsAlphabetic}\\p{IsDigit}]+", " ").trim().split("\\s+");
    for (String token : tokens) {
      if (token.length() > 5) {
        return token;
      }
    }
    if (tokens.length > 0 && !tokens[0].isBlank()) {
      return tokens[0];
    }
    return fallback;
  }

  private record SamplePair(Path docPath, Path vecPath, String baseName) {}

  private record SampleFixture(
      String docId, String queryToken, int originalDimension, float[] projectedVector) {}

  private record VectorInfo(int originalDimension, float[] projected) {}

  private record SampleContext(SamplePair pair, SampleFixture fixture) {}
}
