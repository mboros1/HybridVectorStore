package io.github.mboros1.hvs.search;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.VectorUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Lucene-backed hybrid searcher: BM25 + ANN (HNSW) fused via Reciprocal Rank Fusion.
 */
public final class HybridSearcher implements Closeable {
  private static final Logger LOG = LoggerFactory.getLogger(HybridSearcher.class);
  private static final String FIELD_ID = "id";
  private static final String FIELD_CONTENT = "content";
  private static final String FIELD_VECTOR = "emb";
  private static final String FIELD_PATH_ORD = "path_ord";
  private static final String FIELD_OFFSET = "off";
  private static final String FIELD_LENGTH = "len";
  private static final String FIELD_PREVIEW = "preview";
  private static final int DEFAULT_RRF_K = 60;

  private final Directory directory;
  private final IndexReader reader;
  private final IndexSearcher searcher;
  private final StandardAnalyzer analyzer;
  private final ObjectMapper mapper;
  private final Map<Integer, Path> ordinalToPath;
  private final Map<Integer, FileChannel> channelCache;

  public HybridSearcher(Path indexDir, Path pathsJson) throws IOException {
    Objects.requireNonNull(indexDir, "indexDir");
    Objects.requireNonNull(pathsJson, "pathsJson");
    this.directory = new MMapDirectory(indexDir);
    this.reader = DirectoryReader.open(directory);
    this.searcher = new IndexSearcher(reader);
    this.analyzer = new StandardAnalyzer();
    this.mapper = new ObjectMapper();
    this.ordinalToPath = Collections.unmodifiableMap(loadPaths(pathsJson));
    this.channelCache = new ConcurrentHashMap<>();
    LOG.info(
        "HybridSearcher ready: index={}, docs={}, paths={}",
        indexDir,
        reader.maxDoc(),
        ordinalToPath.size());
  }

  public List<SearchResult> search(SearchRequest request) throws IOException {
    Objects.requireNonNull(request, "request");
    int topK = Math.max(1, request.topK());

    List<RankedHit> bm25Hits = Collections.emptyList();
    if (request.queryText() != null && !request.queryText().isBlank()) {
      bm25Hits = runBm25(request.queryText(), request.bm25TopK());
    }

    List<RankedHit> annHits = Collections.emptyList();
    if (request.queryVector().isPresent()) {
      float[] vector = request.queryVector().get();
      if (request.normalizeVector()) {
        vector = normalize(vector);
      }
      if (vector != null) {
        annHits = runAnn(vector, request.annTopK());
      }
    }

    if (bm25Hits.isEmpty() && annHits.isEmpty()) {
      return List.of();
    }

    Map<Integer, FusionAccumulator> fused = new HashMap<>();
    accumulate(fused, bm25Hits, request.bm25Weight(), "bm25");
    accumulate(fused, annHits, request.annWeight(), "ann");

    List<FusedHit> ordered =
        fused.entrySet().stream()
            .map(e -> new FusedHit(e.getKey(), e.getValue()))
            .sorted(Comparator.comparingDouble(FusedHit::fusedScore).reversed())
            .limit(topK)
            .toList();

    List<SearchResult> results = new ArrayList<>(ordered.size());
    StoredFields storedFields = searcher.storedFields();
    for (FusedHit hit : ordered) {
      Document doc = storedFields.document(hit.docId());
      SearchResult.Builder builder =
          SearchResult.builder()
              .id(doc.get(FIELD_ID))
              .fusedScore(hit.fusedScore())
              .bm25Score(hit.accumulator().bm25Score())
              .annScore(hit.accumulator().annScore())
              .preview(doc.get(FIELD_PREVIEW));

      int pathOrd = numeric(doc, FIELD_PATH_ORD).intValue();
      long offset = numeric(doc, FIELD_OFFSET).longValue();
      int length = numeric(doc, FIELD_LENGTH).intValue();
      Path resolvedPath = resolvePath(pathOrd);
      DocPointer pointer = new DocPointer(pathOrd, resolvedPath, offset, length);
      builder.pointer(pointer);

      if (request.fetchFullDocument()) {
        builder.fullDocument(readDocument(pointer));
      }

      results.add(builder.build());
    }
    return results;
  }

  private void accumulate(
      Map<Integer, FusionAccumulator> fused,
      List<RankedHit> hits,
      double weight,
      String source) {
    if (hits.isEmpty() || weight <= 0d) {
      return;
    }
    for (int i = 0; i < hits.size(); i++) {
      RankedHit hit = hits.get(i);
      FusionAccumulator acc =
          fused.computeIfAbsent(hit.docId(), id -> new FusionAccumulator());
      double add = weight / (DEFAULT_RRF_K + (i + 1));
      acc.add(add);
      if ("bm25".equals(source)) {
        acc.setBm25Score(hit.score());
      } else if ("ann".equals(source)) {
        acc.setAnnScore(hit.score());
      }
    }
  }

  private List<RankedHit> runBm25(String queryText, int topK) throws IOException {
    try {
      QueryParser parser = new QueryParser(FIELD_CONTENT, analyzer);
      Query query = parser.parse(queryText);
      TopDocs docs = searcher.search(query, Math.max(1, topK));
      return toRankedHits(docs);
    } catch (ParseException e) {
      throw new IOException("Failed to parse BM25 query", e);
    }
  }

  private List<RankedHit> runAnn(float[] query, int topK) throws IOException {
    KnnFloatVectorQuery q = new KnnFloatVectorQuery(FIELD_VECTOR, query, Math.max(1, topK));
    TopDocs docs = searcher.search(q, Math.max(1, topK));
    return toRankedHits(docs);
  }

  private List<RankedHit> toRankedHits(TopDocs docs) {
    if (docs == null || docs.scoreDocs == null || docs.scoreDocs.length == 0) {
      return List.of();
    }
    List<RankedHit> hits = new ArrayList<>(docs.scoreDocs.length);
    ScoreDoc[] arr = docs.scoreDocs;
    for (int i = 0; i < arr.length; i++) {
      ScoreDoc sd = arr[i];
      hits.add(new RankedHit(sd.doc, sd.score, i + 1));
    }
    return hits;
  }

  private static Number numeric(Document doc, String field) {
    return doc.getField(field).numericValue();
  }

  private Path resolvePath(int ordinal) throws IOException {
    Path path = ordinalToPath.get(ordinal);
    if (path == null) {
      throw new IOException("missing path for ordinal " + ordinal);
    }
    return path;
  }

  private String readDocument(DocPointer pointer) throws IOException {
    Path path = pointer.path();
    long offset = pointer.offset();
    int length = pointer.length();
    if (length <= 0) {
      return "";
    }
    FileChannel channel = channelFor(pointer.pathOrdinal());
    ByteBuffer buffer = ByteBuffer.allocate(length);
    long position = offset;
    while (buffer.hasRemaining()) {
      int read = channel.read(buffer, position);
      if (read == -1) {
        break;
      }
      position += read;
    }
    buffer.flip();
    return StandardCharsets.UTF_8.decode(buffer).toString();
  }

  private FileChannel channelFor(int ordinal) throws IOException {
    FileChannel existing = channelCache.get(ordinal);
    if (existing != null && existing.isOpen()) {
      return existing;
    }
    synchronized (channelCache) {
      existing = channelCache.get(ordinal);
      if (existing != null && existing.isOpen()) {
        return existing;
      }
      Path path = resolvePath(ordinal);
      FileChannel channel = FileChannel.open(path, StandardOpenOption.READ);
      channelCache.put(ordinal, channel);
      return channel;
    }
  }

  private Map<Integer, Path> loadPaths(Path pathsJson) throws IOException {
    JsonNode node = mapper.readTree(pathsJson.toFile());
    Map<Integer, Path> map = new LinkedHashMap<>();
    Path base = pathsJson.toAbsolutePath().getParent();
    node.fieldNames()
        .forEachRemaining(
            key -> {
              JsonNode value = node.get(key);
              if (value == null || !value.isTextual()) {
                return;
              }
              int ord = Integer.parseInt(key);
              Path raw = Paths.get(value.asText());
              Path resolved = raw.isAbsolute() || base == null ? raw : base.resolve(raw).normalize();
              map.put(ord, resolved);
            });
    return map;
  }

  @Override
  public void close() throws IOException {
    IOException thrown = null;
    for (FileChannel ch : channelCache.values()) {
      try {
        if (ch != null && ch.isOpen()) {
          ch.close();
        }
      } catch (IOException e) {
        if (thrown == null) {
          thrown = e;
        } else {
          thrown.addSuppressed(e);
        }
      }
    }
    channelCache.clear();
    try {
      reader.close();
    } catch (IOException e) {
      if (thrown == null) {
        thrown = e;
      } else {
        thrown.addSuppressed(e);
      }
    }
    try {
      directory.close();
    } catch (IOException e) {
      if (thrown == null) {
        thrown = e;
      } else {
        thrown.addSuppressed(e);
      }
    }
    if (thrown != null) {
      throw thrown;
    }
  }

  private static float[] normalize(float[] vector) {
    double norm = VectorUtil.dotProduct(vector, vector);
    if (norm <= 0d) {
      return null;
    }
    float scale = (float) (1.0d / Math.sqrt(norm));
    float[] copy = vector.clone();
    for (int i = 0; i < copy.length; i++) {
      copy[i] *= scale;
    }
    return copy;
  }

  private record RankedHit(int docId, float score, int rank) {}

  private static final class FusionAccumulator {
    private double fusedScore;
    private Float bm25Score;
    private Float annScore;

    void add(double value) {
      fusedScore += value;
    }

    void setBm25Score(float score) {
      bm25Score = score;
    }

    void setAnnScore(float score) {
      annScore = score;
    }

    Float bm25Score() {
      return bm25Score;
    }

    Float annScore() {
      return annScore;
    }
  }

  private record FusedHit(int docId, FusionAccumulator accumulator) {
    double fusedScore() {
      return accumulator.fusedScore;
    }
  }

  public record DocPointer(int pathOrdinal, Path path, long offset, int length) {}

  public static final class SearchResult {
    private final String id;
    private final double fusedScore;
    private final Float bm25Score;
    private final Float annScore;
    private final String preview;
    private final DocPointer pointer;
    private final String fullDocument;

    private SearchResult(
        String id,
        double fusedScore,
        Float bm25Score,
        Float annScore,
        String preview,
        DocPointer pointer,
        String fullDocument) {
      this.id = id;
      this.fusedScore = fusedScore;
      this.bm25Score = bm25Score;
      this.annScore = annScore;
      this.preview = preview;
      this.pointer = pointer;
      this.fullDocument = fullDocument;
    }

    public static Builder builder() {
      return new Builder();
    }

    public String id() {
      return id;
    }

    public double fusedScore() {
      return fusedScore;
    }

    public Optional<Float> bm25Score() {
      return Optional.ofNullable(bm25Score);
    }

    public Optional<Float> annScore() {
      return Optional.ofNullable(annScore);
    }

    public Optional<String> preview() {
      return Optional.ofNullable(preview);
    }

    public DocPointer pointer() {
      return pointer;
    }

    public Optional<String> fullDocument() {
      return Optional.ofNullable(fullDocument);
    }

    public static final class Builder {
      private String id;
      private double fusedScore;
      private Float bm25Score;
      private Float annScore;
      private String preview;
      private DocPointer pointer;
      private String fullDocument;

      private Builder() {}

      public Builder id(String id) {
        this.id = id;
        return this;
      }

      public Builder fusedScore(double fusedScore) {
        this.fusedScore = fusedScore;
        return this;
      }

      public Builder bm25Score(Float bm25Score) {
        this.bm25Score = bm25Score;
        return this;
      }

      public Builder annScore(Float annScore) {
        this.annScore = annScore;
        return this;
      }

      public Builder preview(String preview) {
        this.preview = preview;
        return this;
      }

      public Builder pointer(DocPointer pointer) {
        this.pointer = pointer;
        return this;
      }

      public Builder fullDocument(String fullDocument) {
        this.fullDocument = fullDocument;
        return this;
      }

      public SearchResult build() {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(pointer, "pointer");
        return new SearchResult(id, fusedScore, bm25Score, annScore, preview, pointer, fullDocument);
      }
    }
  }

  public record SearchRequest(
      String queryText,
      Optional<float[]> queryVector,
      int topK,
      int bm25TopK,
      int annTopK,
      double bm25Weight,
      double annWeight,
      boolean fetchFullDocument,
      boolean normalizeVector) {

    public static Builder builder() {
      return new Builder();
    }

    public static final class Builder {
      private String queryText;
      private float[] queryVector;
      private int topK = 20;
      private int bm25TopK = 100;
      private int annTopK = 100;
      private double bm25Weight = 1.0d;
      private double annWeight = 1.0d;
      private boolean fetchFullDocument = false;
      private boolean normalizeVector = true;

      public Builder text(String queryText) {
        this.queryText = queryText;
        return this;
      }

      public Builder vector(float[] queryVector) {
        this.queryVector = queryVector;
        return this;
      }

      public Builder topK(int topK) {
        this.topK = topK;
        return this;
      }

      public Builder bm25TopK(int bm25TopK) {
        this.bm25TopK = bm25TopK;
        return this;
      }

      public Builder annTopK(int annTopK) {
        this.annTopK = annTopK;
        return this;
      }

      public Builder bm25Weight(double bm25Weight) {
        this.bm25Weight = bm25Weight;
        return this;
      }

      public Builder annWeight(double annWeight) {
        this.annWeight = annWeight;
        return this;
      }

      public Builder fetchFullDocument(boolean fetchFullDocument) {
        this.fetchFullDocument = fetchFullDocument;
        return this;
      }

      public Builder normalizeVector(boolean normalizeVector) {
        this.normalizeVector = normalizeVector;
        return this;
      }

      public SearchRequest build() {
        Optional<float[]> vector =
            queryVector == null ? Optional.empty() : Optional.of(queryVector.clone());
        return new SearchRequest(
            queryText,
            vector,
            topK,
            bm25TopK,
            annTopK,
            bm25Weight,
            annWeight,
            fetchFullDocument,
            normalizeVector);
      }
    }
  }
}
