package io.github.mboros1.hvs.integration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.channels.FileChannel;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
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
    IntegrationRun run = indexSampleShard();
    HybridIndexer.IndexStats stats = run.stats();
    SampleFixture fixture = run.context().fixture();

    assertTrue(stats.docsIndexed() > 0, "Expected at least one document indexed");
    assertEquals(
        stats.docsIndexed(),
        stats.vectorsIndexed(),
        "Vector count should match indexed document count");
    assertEquals(
        fixture.projectedVector().length,
        stats.dimension(),
        "Vector dimension should match sample embedding length");
    assertEquals(fixture.originalDimension(), stats.rawDimension());
    assertEquals(0L, stats.duplicateVectors());
    assertTrue(stats.truncated(), "Expected raw dimension truncation to occur");

    assertTrue(Files.exists(run.manifest()), "manifest.json should be written");
    assertTrue(Files.exists(run.pathsJson()), "paths.json should be written");

    try (HybridSearcher searcher = new HybridSearcher(run.indexDir(), run.pathsJson())) {
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

  @Test
  void manifestAndSidecarMetadataRecorded() throws IOException {
    IntegrationRun run = indexSampleShard();
    HybridIndexer.IndexStats stats = run.stats();
    SampleContext context = run.context();

    long entries = stats.shardResults().get(0).totalLines();
    long expectedSidecarBytes = entries * (Long.BYTES + Integer.BYTES);
    assertEquals(expectedSidecarBytes, Files.size(run.sidecar()));

    JsonNode manifest = MAPPER.readTree(run.manifest().toFile());
    assertEquals(stats.docsIndexed(), manifest.path("docs").asLong());
    assertEquals(stats.vectorsIndexed(), manifest.path("vectors").asLong());
    assertEquals(stats.dimension(), manifest.path("dimension").asInt());
    assertEquals(stats.rawDimension(), manifest.path("rawDimension").asInt());
    assertTrue(manifest.path("vectorTruncated").asBoolean());
    assertEquals(0L, manifest.path("duplicateVectors").asLong());

    JsonNode sidecars = manifest.path("docSidecars");
    assertTrue(sidecars.isArray() && sidecars.size() == 1);
    JsonNode shardNode = sidecars.get(0);
    assertEquals(stats.shardResults().get(0).rawDimension(), shardNode.path("rawDimension").asInt());
    assertTrue(shardNode.path("vectorTruncated").asBoolean());
    assertEquals(stats.shardResults().get(0).duplicateVectors(), shardNode.path("duplicateVectors").asLong());

    Path docPath = context.pair().docPath();
    HybridSearcher.SearchResult topResult;
    try (HybridSearcher searcher = new HybridSearcher(run.indexDir(), run.pathsJson())) {
      List<HybridSearcher.SearchResult> hits =
          searcher.search(
              HybridSearcher.SearchRequest.builder()
                  .vector(context.fixture().projectedVector())
                  .topK(5)
                  .build());
      assertFalse(hits.isEmpty());
      topResult = hits.get(0);
    }

    HybridSearcher.DocPointer pointer = topResult.pointer();
    try (FileChannel ch = FileChannel.open(pointer.path(), StandardOpenOption.READ)) {
      ByteBuffer buf = ByteBuffer.allocate(pointer.length());
      int read = ch.read(buf, pointer.offset());
      assertEquals(pointer.length(), read);
      buf.flip();
      String json = StandardCharsets.UTF_8.decode(buf).toString();
      JsonNode parsed = MAPPER.readTree(json);
      assertEquals(topResult.id(), parsed.path("id").asText());
    }

    JsonNode manifestChecksums = manifest.path("checksums");
    String sidecarRef = shardNode.path("idxPath").asText();
    Path refPath = Paths.get(sidecarRef);
    Path sidecarPath = refPath.isAbsolute() ? refPath : run.outputDir().resolve(refPath).normalize();
    assertEquals(sha256(sidecarPath), manifestChecksums.path(sidecarRef).asText());
  }

  @Test
  void specialCharacterQueryFallsBackToEscapedSearch() throws IOException {
    IntegrationRun run = indexSampleShard();
    String nasty =
        run.context().fixture().queryToken() + " + - && || ! ( ) { } [ ] ^ \" ~ * ? : \\\\ /";
    try (HybridSearcher searcher = new HybridSearcher(run.indexDir(), run.pathsJson())) {
      HybridSearcher.SearchRequest request =
          HybridSearcher.SearchRequest.builder().text(nasty).topK(5).build();
      List<HybridSearcher.SearchResult> hits = searcher.search(request);
      assertFalse(hits.isEmpty(), "Expected fallback parsing to return hits");
    }
  }

  @Test
  void annOnlyQueryReturnsMatches() throws IOException {
    IntegrationRun run = indexSampleShard();
    try (HybridSearcher searcher = new HybridSearcher(run.indexDir(), run.pathsJson())) {
      HybridSearcher.SearchRequest request =
          HybridSearcher.SearchRequest.builder()
              .vector(run.context().fixture().projectedVector())
              .text(null)
              .topK(3)
              .build();
      List<HybridSearcher.SearchResult> hits = searcher.search(request);
      assertFalse(hits.isEmpty(), "ANN-only query should return hits");
    }
  }

  private IntegrationRun indexSampleShard() throws IOException {
    Path samplesDir = Paths.get("samples");
    Assumptions.assumeTrue(
        Files.isDirectory(samplesDir), "samples/ directory must exist for integration test");

    HybridIndexer indexer = new HybridIndexer();
    SampleContext context = locateSampleWithVectors(samplesDir, indexer.maxVectorDimension());
    Assumptions.assumeTrue(context != null, "No matching doc/vec shard pair found under samples/");

    Path runRoot = Files.createTempDirectory(tempDir, "run-");
    Path indexDir = runRoot.resolve("index");
    Path outputDir = runRoot.resolve("out");
    Files.createDirectories(indexDir);
    Files.createDirectories(outputDir);
    Path sidecar = runRoot.resolve(context.pair().baseName() + ".idx");

    HybridIndexer.ShardInput shardInput =
        HybridIndexer.ShardInput.of(context.pair().docPath(), context.pair().vecPath())
            .withSidecar(sidecar);
    HybridIndexer.IndexRequest request =
        new HybridIndexer.IndexRequest(List.of(shardInput), indexDir, outputDir);
    HybridIndexer.IndexStats stats = indexer.index(request);

    Path manifest = outputDir.resolve("manifest.json");
    Path pathsJson = outputDir.resolve("paths.json");
    return new IntegrationRun(context, stats, indexDir, outputDir, manifest, pathsJson, sidecar);
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

  private record IntegrationRun(
      SampleContext context,
      HybridIndexer.IndexStats stats,
      Path indexDir,
      Path outputDir,
      Path manifest,
      Path pathsJson,
      Path sidecar) {}

  private static String sha256(Path path) throws IOException {
    MessageDigest digest;
    try {
      digest = MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException("Missing SHA-256 provider", e);
    }
    try (InputStream in = Files.newInputStream(path)) {
      byte[] buffer = new byte[8192];
      int read;
      while ((read = in.read(buffer)) != -1) {
        digest.update(buffer, 0, read);
      }
    }
    byte[] hash = digest.digest();
    StringBuilder sb = new StringBuilder(hash.length * 2);
    for (byte b : hash) {
      sb.append(String.format("%02x", b));
    }
    return sb.toString();
  }
}
