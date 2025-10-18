package io.github.mboros1.hvs.index;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.github.mboros1.hvs.search.HybridSearcher;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class HybridIndexerTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  @TempDir Path tempDir;

  @Test
  void indexAndSearchSampleCorpusSlice() throws IOException {
    Path sampleDir = Paths.get("samples", "embedded_docs");
    Assumptions.assumeTrue(
        Files.isDirectory(sampleDir),
        "Sample corpus directory samples/embedded_docs must exist to run this test");

    Path sampleFile;
    try (Stream<Path> stream = Files.list(sampleDir)) {
      sampleFile =
          stream
              .filter(Files::isRegularFile)
              .filter(p -> p.toString().endsWith(".json"))
              .findFirst()
              .orElseThrow(
                  () ->
                      new IllegalStateException(
                          "No sample JSON files found under " + sampleDir.toAbsolutePath()));
    }

    JsonNode root = MAPPER.readTree(sampleFile.toFile());
    if (!root.isArray()) {
      throw new IllegalStateException("Expected array in " + sampleFile);
    }
    ArrayNode arr = (ArrayNode) root;

    Path docsNdjson = tempDir.resolve("slice.doc.ndjson");
    Path vecNdjson = tempDir.resolve("slice.vec.ndjson");
    List<String> ids = new ArrayList<>();
    List<float[]> vectors = new ArrayList<>();
    List<String> texts = new ArrayList<>();

    int maxDocs = Math.min(12, arr.size());
    try (BufferedWriter docWriter = Files.newBufferedWriter(docsNdjson, StandardCharsets.UTF_8);
        BufferedWriter vecWriter = Files.newBufferedWriter(vecNdjson, StandardCharsets.UTF_8)) {
      for (int i = 0; i < arr.size() && ids.size() < maxDocs; i++) {
        JsonNode node = arr.get(i);
        String id = node.path("id").asText(null);
        String text = node.path("text").asText(node.path("content").asText(""));
        JsonNode metadata = node.path("metadata");
        JsonNode embeddingNode = metadata.path("embedding");
        if (id == null || id.isBlank() || text.isBlank() || !embeddingNode.isArray()) {
          continue;
        }
        ArrayNode embeddingArray = (ArrayNode) embeddingNode;
        if (embeddingArray.isEmpty()) {
          continue;
        }
        ObjectNode doc = MAPPER.createObjectNode();
        doc.put("id", id);
        doc.put("text", text);
        ObjectNode metaCopy =
            metadata.isObject() ? metadata.deepCopy() : MAPPER.createObjectNode();
        metaCopy.remove("embedding");
        doc.set("metadata", metaCopy);
        String docLine = MAPPER.writeValueAsString(doc);
        docWriter.write(docLine);
        docWriter.newLine();

        ObjectNode vec = MAPPER.createObjectNode();
        vec.put("id", id);
        ArrayNode embCopy = MAPPER.createArrayNode();
        float[] vector = new float[embeddingArray.size()];
        for (int j = 0; j < embeddingArray.size(); j++) {
          float value = embeddingArray.get(j).floatValue();
          embCopy.add(value);
          vector[j] = value;
        }
        vec.set("embeddings", embCopy);
        vecWriter.write(MAPPER.writeValueAsString(vec));
        vecWriter.newLine();

        ids.add(id);
        vectors.add(vector);
        texts.add(text);
      }
    }

    assertFalse(ids.isEmpty(), "Sample slice should include at least one document");

    Path indexDir = tempDir.resolve("index");
    Path outputDir = tempDir.resolve("out");
    HybridIndexer indexer = new HybridIndexer();
    HybridIndexer.IndexRequest request =
        new HybridIndexer.IndexRequest(
            List.of(HybridIndexer.ShardInput.of(docsNdjson, vecNdjson)), indexDir, outputDir);
    HybridIndexer.IndexStats stats = indexer.index(request);

    assertEquals(ids.size(), stats.docsIndexed());
    assertEquals(ids.size(), stats.vectorsIndexed());
    assertTrue(stats.dimension() > 0, "Vector dimension should be detected");

    Path manifest = outputDir.resolve("manifest.json");
    assertTrue(Files.exists(manifest), "manifest.json should exist");

    Path pathsJson = outputDir.resolve("paths.json");
    try (HybridSearcher searcher = new HybridSearcher(indexDir, pathsJson)) {
      float[] queryVector = vectors.get(0);
      String textQuery = texts.get(0).split("\\s+")[0];

      HybridSearcher.SearchRequest searchRequest =
          HybridSearcher.SearchRequest.builder()
              .text(textQuery)
              .vector(queryVector)
              .fetchFullDocument(true)
              .topK(5)
              .bm25TopK(10)
              .annTopK(10)
              .build();

      List<HybridSearcher.SearchResult> results = searcher.search(searchRequest);
      assertFalse(results.isEmpty(), "Hybrid search should return at least one hit");
      HybridSearcher.SearchResult top = results.get(0);
      assertEquals(ids.get(0), top.id(), "First hit should match the query document");
      assertTrue(
          top.fullDocument().isPresent(),
          "Full document payload expected when fetchFullDocument=true");
      String fullDoc = top.fullDocument().get();
      assertTrue(
          fullDoc.contains(ids.get(0)),
          "Full document should contain original identifier");
      assertTrue(
          fullDoc.contains(textQuery),
          "Full document should contain text snippet from the source sample");
    }
  }
}
