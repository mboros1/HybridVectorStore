package io.github.mboros1.hvs.index;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.Version;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Hybrid indexer that wires NDJSON documents + float vectors into a Lucene index while emitting
 * sidecar metadata required for fast document retrieval.
 */
public final class HybridIndexer {
  private static final Logger LOG = LoggerFactory.getLogger(HybridIndexer.class);
  public static final int DEFAULT_MAX_VECTOR_DIMENSION = 1024;
  private static final String FIELD_ID = "id";
  private static final String FIELD_CONTENT = "content";
  private static final String FIELD_VECTOR = "emb";
  private static final String FIELD_PATH_ORD = "path_ord";
  private static final String FIELD_OFFSET = "off";
  private static final String FIELD_LENGTH = "len";
  private static final String FIELD_PREVIEW = "preview";

  private final ObjectMapper mapper;
  private final int maxVectorDimension;

  public HybridIndexer() {
    this(DEFAULT_MAX_VECTOR_DIMENSION);
  }

  public HybridIndexer(int maxVectorDimension) {
    this.mapper = new ObjectMapper(new JsonFactory());
    if (maxVectorDimension <= 0) {
      throw new IllegalArgumentException("maxVectorDimension must be positive");
    }
    this.maxVectorDimension = maxVectorDimension;
  }

  public IndexStats index(IndexRequest request) throws IOException {
    Objects.requireNonNull(request, "request");
    if (request.shards().isEmpty()) {
      throw new IllegalArgumentException("No shards provided");
    }
    Files.createDirectories(request.indexDir());
    Files.createDirectories(request.outputDir());

    PathsRegistry pathsRegistry = new PathsRegistry();

    IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer());
    config.setOpenMode(OpenMode.CREATE);

    long totalDocs = 0;
    long indexedDocs = 0;
    long missingVectors = 0;
    long vectorsIngested = 0;
    int dimension = -1;

    List<ShardResult> shardResults = new ArrayList<>();

    try (Directory dir = new MMapDirectory(request.indexDir());
         IndexWriter writer = new IndexWriter(dir, config)) {
      for (ShardInput shard : request.shards()) {
        ShardProcessor processor = new ShardProcessor(shard, pathsRegistry, writer);
        ShardResult result = processor.process();
        totalDocs += result.totalLines();
        indexedDocs += result.indexedDocs();
        missingVectors += result.missingVectors();
        vectorsIngested += result.vectorsConsumed();
        if (dimension < 0) {
          dimension = result.dimension();
        } else if (result.dimension() > 0 && result.dimension() != dimension) {
          throw new IOException(
              "Vector dimension mismatch: expected " + dimension + " got " + result.dimension());
        }
        shardResults.add(result);
      }
      writer.commit();
    }

    Path pathsJson = request.outputDir().resolve("paths.json");
    pathsRegistry.write(mapper, pathsJson);

    Path manifest = request.outputDir().resolve("manifest.json");
    writeManifest(
        manifest, pathsJson, request.indexDir(), shardResults, indexedDocs, vectorsIngested, dimension);

    return new IndexStats(
        indexedDocs,
        totalDocs - indexedDocs,
        missingVectors,
        vectorsIngested,
        dimension,
        request.indexDir(),
        manifest,
        List.copyOf(shardResults));
  }

  private void writeManifest(
      Path manifestPath,
      Path pathsJson,
      Path indexDir,
      List<ShardResult> shardResults,
      long docsIndexed,
      long vectorsIndexed,
      int dimension)
      throws IOException {
    Path parentDir = manifestPath.getParent();
    if (parentDir != null) {
      Files.createDirectories(parentDir);
    }
    ObjectNode root = mapper.createObjectNode();
    root.put("docs", docsIndexed);
    root.put("vectors", vectorsIndexed);
    root.put("dimension", dimension);
    root.put("dtype", "f32");
    Path manifestDir = manifestPath.getParent();
    String pathsRef = relativizeOrAbsolute(manifestDir, pathsJson);
    String indexRef = relativizeOrAbsolute(manifestDir, indexDir);
    root.put("paths", pathsRef);
    root.put("luceneIndex", indexRef);
    root.put("luceneVersion", Version.LATEST.toString());
    root.put("generatedAt", Instant.now().toString());

    ObjectNode checksums = root.putObject("checksums");
    checksums.put(pathsRef, sha256(pathsJson));
    ArrayNode sidecars = root.putArray("docSidecars");
    for (ShardResult shard : shardResults) {
      ObjectNode node = sidecars.addObject();
      node.put("docPath", shard.docPath().toString());
      String idxRef = relativizeOrAbsolute(manifestDir, shard.docIndexPath());
      node.put("idxPath", idxRef);
      node.put("entries", shard.totalLines());
      String checksum = sha256(shard.docIndexPath());
      node.put("checksum", checksum);
      checksums.put(idxRef, checksum);
    }

    mapper.writerWithDefaultPrettyPrinter().writeValue(manifestPath.toFile(), root);
  }

  private static String relativizeOrAbsolute(Path base, Path target) {
    if (base == null) {
      return target.toString();
    }
    try {
      return base.relativize(target).toString();
    } catch (IllegalArgumentException e) {
      return target.toString();
    }
  }

  private static String sha256(Path file) throws IOException {
    MessageDigest digest;
    try {
      digest = MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException("Missing SHA-256 provider", e);
    }
    try (InputStream in = Files.newInputStream(file)) {
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

  private final class ShardProcessor {
    private final ShardInput shard;
    private final PathsRegistry registry;
    private final IndexWriter writer;

    ShardProcessor(ShardInput shard, PathsRegistry registry, IndexWriter writer) {
      this.shard = shard;
      this.registry = registry;
      this.writer = writer;
    }

    ShardResult process() throws IOException {
      Path docPath = shard.docPath();
      Path vectorPath = shard.vectorPath();
      Path idxPath = shard.sidecarPath().orElse(defaultIdxPath(docPath));

      Path idxParent = idxPath.getParent();
      if (idxParent != null) {
        Files.createDirectories(idxParent);
      }
      int pathOrd = registry.register(docPath);

      VectorData vectors = loadVectors(vectorPath);
      long totalLines = 0;
      long indexedDocs = 0;
      long missingVectors = 0;

      try (DocLineReader reader = new DocLineReader(docPath);
           DocSidecarWriter idxWriter = new DocSidecarWriter(idxPath)) {
        Optional<DocLine> maybeLine;
        while ((maybeLine = reader.next()).isPresent()) {
          DocLine line = maybeLine.get();
          totalLines++;
          idxWriter.append(line.offset(), line.length());
          JsonNode node = line.parse(mapper);
          String id = text(node.get("id"));
          if (id == null || id.isBlank()) {
            LOG.warn("Skipping doc without id at line {} in {}", totalLines, docPath);
            continue;
          }
          float[] vector = vectors.consume(id);
          if (vector == null) {
            missingVectors++;
            continue;
          }
          String text = extractText(node);
          Document doc = new Document();
          doc.add(new StringField(FIELD_ID, id, Field.Store.YES));
          if (!text.isBlank()) {
            doc.add(new TextField(FIELD_CONTENT, text, Field.Store.NO));
            doc.add(new StoredField(
                FIELD_PREVIEW, text.length() > 256 ? text.substring(0, 256) : text));
          }
          doc.add(
              new KnnFloatVectorField(
                  FIELD_VECTOR, vector, VectorSimilarityFunction.DOT_PRODUCT));
          doc.add(new NumericDocValuesField(FIELD_PATH_ORD, pathOrd));
          doc.add(new NumericDocValuesField(FIELD_OFFSET, line.offset()));
          doc.add(new NumericDocValuesField(FIELD_LENGTH, line.length()));
          doc.add(new StoredField(FIELD_PATH_ORD, pathOrd));
          doc.add(new StoredField(FIELD_OFFSET, line.offset()));
          doc.add(new StoredField(FIELD_LENGTH, line.length()));
          writer.addDocument(doc);
          indexedDocs++;
        }
      }

      if (vectors.remainingCount() > 0) {
        LOG.warn(
            "Shard {} vectors without matching docs: {} (sample ids: {})",
            vectorPath,
            vectors.remainingCount(),
            vectors.sampleIds(5));
      }

      return new ShardResult(
          docPath,
          idxPath,
          totalLines,
          indexedDocs,
          missingVectors,
          vectors.consumedCount(),
          vectors.dimension());
    }
  }

  private static Path defaultIdxPath(Path docPath) {
    String name = docPath.getFileName().toString();
    String idxName;
    if (name.endsWith(".doc.ndjson")) {
      idxName = name.substring(0, name.length() - ".doc.ndjson".length()) + ".doc.idx";
    } else {
      idxName = name + ".idx";
    }
    return docPath.resolveSibling(idxName);
  }

  private VectorData loadVectors(Path vectorPath) throws IOException {
    Map<String, float[]> vectors = new HashMap<>();
    int dimension = -1;
    int rawDimension = -1;
    AtomicBoolean truncated = new AtomicBoolean(false);
    long count = 0;
    try (DocLineReader reader = new DocLineReader(vectorPath)) {
      Optional<DocLine> maybeLine;
      while ((maybeLine = reader.next()).isPresent()) {
        DocLine line = maybeLine.get();
        JsonNode node = line.parse(mapper);
        String id = text(node.get("id"));
        if (id == null || id.isBlank()) {
          continue;
        }
        JsonNode vecNode = node.path("embeddings");
        if (vecNode.isMissingNode()) {
          vecNode = node.path("embedding");
        }
        if (!vecNode.isArray()) {
          continue;
        }
        int len = vecNode.size();
        if (rawDimension < 0) {
          rawDimension = len;
        } else if (len != rawDimension) {
          throw new IOException(
              "Vector dimension mismatch in "
                  + vectorPath
                  + " for id "
                  + id
                  + ": expected "
                  + rawDimension
                  + " len "
                  + len);
        }
        int effectiveLen = Math.min(len, maxVectorDimension);
        if (effectiveLen < len) {
          truncated.set(true);
        }
        if (dimension < 0) {
          dimension = effectiveLen;
        } else if (effectiveLen != dimension) {
          throw new IOException(
              "Vector dimension mismatch in "
                  + vectorPath
                  + " for id "
                  + id
                  + ": expected "
                  + dimension
                  + " len "
                  + effectiveLen);
        }
        float[] vector = new float[effectiveLen];
        double norm = 0d;
        for (int i = 0; i < len; i++) {
          double v = vecNode.get(i).asDouble();
          if (i < effectiveLen) {
            vector[i] = (float) v;
            norm += v * v;
          }
        }
        if (norm == 0d) {
          continue;
        }
        float scale = (float) (1.0d / Math.sqrt(norm));
        for (int i = 0; i < effectiveLen; i++) {
          vector[i] *= scale;
        }
        vectors.put(id, vector);
        count++;
      }
    }
    if (truncated.get()) {
      LOG.warn(
          "Detected vector dimension {} in {} exceeding maximum {}; truncating to {}",
          rawDimension,
          vectorPath,
          maxVectorDimension,
          dimension);
    }
    return new VectorData(vectors, dimension, count);
  }

  private static String extractText(JsonNode node) {
    String text = text(node.get("text"));
    if (text == null || text.isBlank()) {
      text = text(node.get("content"));
    }
    if (text == null) {
      return "";
    }
    return text;
  }

  private static String text(JsonNode node) {
    return node != null && !node.isNull() ? node.asText() : null;
  }

  private static final class PathsRegistry {
    private final Map<Path, Integer> ordinals = new LinkedHashMap<>();

    int register(Path path) {
      Path normalized = path.toAbsolutePath().normalize();
      return ordinals.computeIfAbsent(normalized, p -> ordinals.size());
    }

    void write(ObjectMapper mapper, Path output) throws IOException {
      Path parent = output.getParent();
      if (parent != null) {
        Files.createDirectories(parent);
      }
      ObjectNode root = mapper.createObjectNode();
      for (Map.Entry<Path, Integer> entry : ordinals.entrySet()) {
        root.put(String.valueOf(entry.getValue()), entry.getKey().toString());
      }
      mapper.writerWithDefaultPrettyPrinter().writeValue(output.toFile(), root);
    }
  }

  private record VectorData(Map<String, float[]> vectors, int dimension, long count) {
    float[] consume(String id) {
      return vectors.remove(id);
    }

    long consumedCount() {
      return count - vectors.size();
    }

    long remainingCount() {
      return vectors.size();
    }

    List<String> sampleIds(int limit) {
      List<String> sample = new ArrayList<>(Math.min(limit, vectors.size()));
      int i = 0;
      for (String id : vectors.keySet()) {
        sample.add(id);
        if (++i >= limit) {
          break;
        }
      }
      return sample;
    }
  }

  private static final class DocSidecarWriter implements AutoCloseable {
    private final FileChannel channel;
    private final ByteBuffer buffer;

    DocSidecarWriter(Path target) throws IOException {
      this.channel =
          FileChannel.open(
              target,
              StandardOpenOption.CREATE,
              StandardOpenOption.TRUNCATE_EXISTING,
              StandardOpenOption.WRITE);
      this.buffer = ByteBuffer.allocateDirect(12).order(ByteOrder.LITTLE_ENDIAN);
    }

    void append(long offset, int length) throws IOException {
      buffer.clear();
      buffer.putLong(offset);
      buffer.putInt(length);
      buffer.flip();
      channel.write(buffer);
    }

    @Override
    public void close() throws IOException {
      channel.close();
    }
  }

  private static final class DocLineReader implements AutoCloseable {
    private final InputStream in;
    private long offset = 0L;

    DocLineReader(Path path) throws IOException {
      this.in = Files.newInputStream(path);
    }

    Optional<DocLine> next() throws IOException {
      ByteArrayOutputStream buffer = new ByteArrayOutputStream();
      long lineOffset = offset;
      boolean sawData = false;
      while (true) {
        int b = in.read();
        if (b == -1) {
          break;
        }
        sawData = true;
        buffer.write(b);
        offset++;
        if (b == '\n') {
          break;
        }
      }
      if (!sawData && buffer.size() == 0) {
        return Optional.empty();
      }
      byte[] bytes = buffer.toByteArray();
      return Optional.of(new DocLine(lineOffset, bytes.length, bytes));
    }

    @Override
    public void close() throws IOException {
      in.close();
    }
  }

  private record DocLine(long offset, int length, byte[] bytes) {
    JsonNode parse(ObjectMapper mapper) throws IOException {
      int jsonLen = trimmedLength();
      return mapper.readTree(new ByteArrayInputStream(bytes, 0, jsonLen));
    }

    int trimmedLength() {
      int jsonLen = length;
      while (jsonLen > 0) {
        byte b = bytes[jsonLen - 1];
        if (b == '\n' || b == '\r') {
          jsonLen--;
        } else {
          break;
        }
      }
      return jsonLen;
    }
  }

  public record IndexRequest(List<ShardInput> shards, Path indexDir, Path outputDir) {
    public IndexRequest {
      Objects.requireNonNull(shards, "shards");
      Objects.requireNonNull(indexDir, "indexDir");
      Objects.requireNonNull(outputDir, "outputDir");
    }
  }

  public record ShardInput(Path docPath, Path vectorPath, Optional<Path> sidecarPath) {
    public ShardInput {
      Objects.requireNonNull(docPath, "docPath");
      Objects.requireNonNull(vectorPath, "vectorPath");
      Objects.requireNonNull(sidecarPath, "sidecarPath");
    }

    public static ShardInput of(Path docPath, Path vectorPath) {
      return new ShardInput(docPath, vectorPath, Optional.empty());
    }

    public ShardInput withSidecar(Path sidecarPath) {
      return new ShardInput(docPath, vectorPath, Optional.of(sidecarPath));
    }
  }

  public record IndexStats(
      long docsIndexed,
      long docsSkipped,
      long missingVectors,
      long vectorsIndexed,
      int dimension,
      Path indexDir,
      Path manifestPath,
      List<ShardResult> shardResults) {}

  public record ShardResult(
      Path docPath,
      Path docIndexPath,
      long totalLines,
      long indexedDocs,
      long missingVectors,
      long vectorsConsumed,
      int dimension) {}

  public int maxVectorDimension() {
    return maxVectorDimension;
  }
}
