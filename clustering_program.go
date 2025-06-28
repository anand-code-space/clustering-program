package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Record represents a data record with ID and description
type Record struct {
	ID          string `json:"id"`
	Description string `json:"description"`
}

// Cluster represents a group of similar records
type Cluster struct {
	ID       int      `json:"id"`
	Centroid []float64 `json:"-"`
	Records  []Record `json:"records"`
	Keywords []string `json:"keywords"`
}

// TFIDFVectorizer handles text vectorization
type TFIDFVectorizer struct {
	Vocabulary map[string]int
	IDF        []float64
	DocCount   int
}

// ClusteringEngine main clustering engine
type ClusteringEngine struct {
	Records    []Record
	Vectorizer *TFIDFVectorizer
	Clusters   []Cluster
	K          int
}

// NewClusteringEngine creates a new clustering engine
func NewClusteringEngine(k int) *ClusteringEngine {
	return &ClusteringEngine{
		Records:    make([]Record, 0),
		Vectorizer: &TFIDFVectorizer{Vocabulary: make(map[string]int)},
		Clusters:   make([]Cluster, 0),
		K:          k,
	}
}

// LoadFromCSV loads records from a CSV file
func (ce *ClusteringEngine) LoadFromCSV(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	// Skip header if present
	if _, err := reader.Read(); err != nil {
		return err
	}

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		if len(record) >= 2 {
			ce.Records = append(ce.Records, Record{
				ID:          record[0],
				Description: record[1],
			})
		}
	}

	fmt.Printf("Loaded %d records from %s\n", len(ce.Records), filename)
	return nil
}

// GenerateSampleData creates sample data for testing
func (ce *ClusteringEngine) GenerateSampleData(count int) {
	rand.Seed(time.Now().UnixNano())
	
	topics := [][]string{
		{"technology", "software", "computer", "programming", "development", "application", "system", "digital"},
		{"health", "medical", "doctor", "hospital", "treatment", "medicine", "patient", "care"},
		{"education", "school", "student", "learning", "teaching", "university", "course", "academic"},
		{"business", "finance", "money", "investment", "company", "market", "sales", "profit"},
		{"sports", "football", "basketball", "game", "team", "player", "competition", "athletic"},
		{"food", "restaurant", "cooking", "recipe", "meal", "ingredient", "kitchen", "dining"},
		{"travel", "vacation", "hotel", "flight", "destination", "tourism", "journey", "adventure"},
		{"entertainment", "movie", "music", "concert", "show", "performance", "artist", "cinema"},
	}

	for i := 0; i < count; i++ {
		topic := topics[rand.Intn(len(topics))]
		words := make([]string, 3+rand.Intn(5))
		
		for j := range words {
			words[j] = topic[rand.Intn(len(topic))]
		}
		
		description := strings.Join(words, " ")
		ce.Records = append(ce.Records, Record{
			ID:          fmt.Sprintf("record_%d", i+1),
			Description: description,
		})
	}
	
	fmt.Printf("Generated %d sample records\n", count)
}

// PreprocessText cleans and tokenizes text
func (ce *ClusteringEngine) PreprocessText(text string) []string {
	// Convert to lowercase
	text = strings.ToLower(text)
	
	// Remove non-alphanumeric characters except spaces
	reg := regexp.MustCompile(`[^a-z0-9\s]`)
	text = reg.ReplaceAllString(text, "")
	
	// Split into words
	words := strings.Fields(text)
	
	// Remove common stop words
	stopWords := map[string]bool{
		"the": true, "a": true, "an": true, "and": true, "or": true,
		"but": true, "in": true, "on": true, "at": true, "to": true,
		"for": true, "of": true, "with": true, "by": true, "is": true,
		"are": true, "was": true, "were": true, "be": true, "been": true,
		"have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "could": true, "should": true,
	}
	
	filtered := make([]string, 0)
	for _, word := range words {
		if len(word) > 2 && !stopWords[word] {
			filtered = append(filtered, word)
		}
	}
	
	return filtered
}

// BuildVocabulary creates vocabulary and calculates IDF values
func (ce *ClusteringEngine) BuildVocabulary() {
	fmt.Println("Building vocabulary...")
	
	wordDocCount := make(map[string]int)
	ce.Vectorizer.DocCount = len(ce.Records)
	
	// Count document frequency for each word
	for _, record := range ce.Records {
		words := ce.PreprocessText(record.Description)
		wordSet := make(map[string]bool)
		
		for _, word := range words {
			wordSet[word] = true
		}
		
		for word := range wordSet {
			wordDocCount[word]++
		}
	}
	
	// Build vocabulary (only words appearing in at least 2 documents)
	vocabIndex := 0
	for word, count := range wordDocCount {
		if count >= 2 {
			ce.Vectorizer.Vocabulary[word] = vocabIndex
			vocabIndex++
		}
	}
	
	// Calculate IDF values
	ce.Vectorizer.IDF = make([]float64, len(ce.Vectorizer.Vocabulary))
	for word, index := range ce.Vectorizer.Vocabulary {
		df := float64(wordDocCount[word])
		ce.Vectorizer.IDF[index] = math.Log(float64(ce.Vectorizer.DocCount) / df)
	}
	
	fmt.Printf("Built vocabulary with %d terms\n", len(ce.Vectorizer.Vocabulary))
}

// VectorizeDocument converts text to TF-IDF vector
func (ce *ClusteringEngine) VectorizeDocument(text string) []float64 {
	words := ce.PreprocessText(text)
	vector := make([]float64, len(ce.Vectorizer.Vocabulary))
	
	// Calculate term frequency
	termCount := make(map[string]int)
	for _, word := range words {
		termCount[word]++
	}
	
	// Calculate TF-IDF
	for word, count := range termCount {
		if index, exists := ce.Vectorizer.Vocabulary[word]; exists {
			tf := float64(count) / float64(len(words))
			vector[index] = tf * ce.Vectorizer.IDF[index]
		}
	}
	
	return vector
}

// CosineSimilarity calculates cosine similarity between two vectors
func CosineSimilarity(v1, v2 []float64) float64 {
	if len(v1) != len(v2) {
		return 0
	}
	
	dotProduct := 0.0
	norm1 := 0.0
	norm2 := 0.0
	
	for i := range v1 {
		dotProduct += v1[i] * v2[i]
		norm1 += v1[i] * v1[i]
		norm2 += v2[i] * v2[i]
	}
	
	if norm1 == 0 || norm2 == 0 {
		return 0
	}
	
	return dotProduct / (math.Sqrt(norm1) * math.Sqrt(norm2))
}

// EuclideanDistance calculates Euclidean distance between vectors
func EuclideanDistance(v1, v2 []float64) float64 {
	if len(v1) != len(v2) {
		return math.Inf(1)
	}
	
	sum := 0.0
	for i := range v1 {
		diff := v1[i] - v2[i]
		sum += diff * diff
	}
	
	return math.Sqrt(sum)
}

// KMeansCluster performs K-means clustering
func (ce *ClusteringEngine) KMeansCluster() {
	fmt.Printf("Starting K-means clustering with k=%d...\n", ce.K)
	
	// Vectorize all documents
	vectors := make([][]float64, len(ce.Records))
	for i, record := range ce.Records {
		vectors[i] = ce.VectorizeDocument(record.Description)
	}
	
	// Initialize centroids randomly
	centroids := make([][]float64, ce.K)
	for i := range centroids {
		centroids[i] = make([]float64, len(ce.Vectorizer.Vocabulary))
		for j := range centroids[i] {
			centroids[i][j] = rand.Float64()
		}
	}
	
	maxIterations := 100
	tolerance := 1e-4
	
	for iteration := 0; iteration < maxIterations; iteration++ {
		// Assign points to clusters
		assignments := make([]int, len(vectors))
		for i, vector := range vectors {
			minDistance := math.Inf(1)
			bestCluster := 0
			
			for j, centroid := range centroids {
				distance := EuclideanDistance(vector, centroid)
				if distance < minDistance {
					minDistance = distance
					bestCluster = j
				}
			}
			assignments[i] = bestCluster
		}
		
		// Update centroids
		newCentroids := make([][]float64, ce.K)
		clusterCounts := make([]int, ce.K)
		
		for i := range newCentroids {
			newCentroids[i] = make([]float64, len(ce.Vectorizer.Vocabulary))
		}
		
		for i, vector := range vectors {
			cluster := assignments[i]
			clusterCounts[cluster]++
			for j, val := range vector {
				newCentroids[cluster][j] += val
			}
		}
		
		// Average the centroids
		for i := range newCentroids {
			if clusterCounts[i] > 0 {
				for j := range newCentroids[i] {
					newCentroids[i][j] /= float64(clusterCounts[i])
				}
			}
		}
		
		// Check for convergence
		converged := true
		for i := range centroids {
			if EuclideanDistance(centroids[i], newCentroids[i]) > tolerance {
				converged = false
				break
			}
		}
		
		centroids = newCentroids
		
		if converged {
			fmt.Printf("Converged after %d iterations\n", iteration+1)
			break
		}
	}
	
	// Create final clusters
	ce.Clusters = make([]Cluster, ce.K)
	for i := range ce.Clusters {
		ce.Clusters[i] = Cluster{
			ID:       i,
			Centroid: centroids[i],
			Records:  make([]Record, 0),
		}
	}
	
	// Assign records to final clusters
	for i, vector := range vectors {
		minDistance := math.Inf(1)
		bestCluster := 0
		
		for j, centroid := range centroids {
			distance := EuclideanDistance(vector, centroid)
			if distance < minDistance {
				minDistance = distance
				bestCluster = j
			}
		}
		
		ce.Clusters[bestCluster].Records = append(ce.Clusters[bestCluster].Records, ce.Records[i])
	}
	
	// Generate keywords for each cluster
	ce.GenerateClusterKeywords()
	
	fmt.Println("Clustering completed!")
}

// GenerateClusterKeywords extracts representative keywords for each cluster
func (ce *ClusteringEngine) GenerateClusterKeywords() {
	for i := range ce.Clusters {
		wordFreq := make(map[string]int)
		
		// Count word frequencies in cluster
		for _, record := range ce.Clusters[i].Records {
			words := ce.PreprocessText(record.Description)
			for _, word := range words {
				if _, exists := ce.Vectorizer.Vocabulary[word]; exists {
					wordFreq[word]++
				}
			}
		}
		
		// Sort words by frequency
		type wordCount struct {
			word  string
			count int
		}
		
		var wordCounts []wordCount
		for word, count := range wordFreq {
			wordCounts = append(wordCounts, wordCount{word, count})
		}
		
		sort.Slice(wordCounts, func(a, b int) bool {
			return wordCounts[a].count > wordCounts[b].count
		})
		
		// Take top 5 keywords
		maxKeywords := 5
		if len(wordCounts) < maxKeywords {
			maxKeywords = len(wordCounts)
		}
		
		ce.Clusters[i].Keywords = make([]string, maxKeywords)
		for j := 0; j < maxKeywords; j++ {
			ce.Clusters[i].Keywords[j] = wordCounts[j].word
		}
	}
}

// PrintResults displays clustering results
func (ce *ClusteringEngine) PrintResults() {
	fmt.Println("\n=== CLUSTERING RESULTS ===")
	fmt.Printf("Total records: %d\n", len(ce.Records))
	fmt.Printf("Number of clusters: %d\n", ce.K)
	fmt.Println()
	
	for _, cluster := range ce.Clusters {
		fmt.Printf("Cluster %d (%d records)\n", cluster.ID, len(cluster.Records))
		fmt.Printf("Keywords: %s\n", strings.Join(cluster.Keywords, ", "))
		
		// Show first few records as examples
		maxExamples := 3
		if len(cluster.Records) < maxExamples {
			maxExamples = len(cluster.Records)
		}
		
		fmt.Println("Examples:")
		for i := 0; i < maxExamples; i++ {
			fmt.Printf("  - %s: %s\n", cluster.Records[i].ID, cluster.Records[i].Description)
		}
		fmt.Println()
	}
}

// SaveResults saves clustering results to JSON file
func (ce *ClusteringEngine) SaveResults(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(ce.Clusters)
}

// AnalyzeClusterQuality provides clustering quality metrics
func (ce *ClusteringEngine) AnalyzeClusterQuality() {
	fmt.Println("\n=== CLUSTER QUALITY ANALYSIS ===")
	
	totalRecords := len(ce.Records)
	nonEmptyClusters := 0
	
	for _, cluster := range ce.Clusters {
		if len(cluster.Records) > 0 {
			nonEmptyClusters++
		}
	}
	
	fmt.Printf("Non-empty clusters: %d/%d\n", nonEmptyClusters, ce.K)
	fmt.Printf("Average cluster size: %.2f\n", float64(totalRecords)/float64(nonEmptyClusters))
	
	// Calculate cluster size distribution
	sizes := make([]int, len(ce.Clusters))
	for i, cluster := range ce.Clusters {
		sizes[i] = len(cluster.Records)
	}
	
	sort.Ints(sizes)
	fmt.Printf("Cluster size range: %d - %d\n", sizes[0], sizes[len(sizes)-1])
}

func main() {
	fmt.Println("Text Clustering Program")
	fmt.Println("======================")
	
	// Get clustering parameters
	var numClusters int
	var dataSource string
	
	fmt.Print("Enter number of clusters (default 8): ")
	reader := bufio.NewReader(os.Stdin)
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)
	
	if input == "" {
		numClusters = 8
	} else {
		var err error
		numClusters, err = strconv.Atoi(input)
		if err != nil || numClusters <= 0 {
			numClusters = 8
		}
	}
	
	fmt.Print("Data source (1: Generate sample data, 2: Load from CSV): ")
	choice, _ := reader.ReadString('\n')
	choice = strings.TrimSpace(choice)
	
	// Create clustering engine
	engine := NewClusteringEngine(numClusters)
	
	if choice == "2" {
		fmt.Print("Enter CSV filename: ")
		filename, _ := reader.ReadString('\n')
		filename = strings.TrimSpace(filename)
		
		if err := engine.LoadFromCSV(filename); err != nil {
			log.Printf("Error loading CSV: %v\n", err)
			fmt.Println("Falling back to sample data generation...")
			dataSource = "sample"
		} else {
			dataSource = "csv"
		}
	} else {
		dataSource = "sample"
	}
	
	if dataSource == "sample" {
		var numRecords int
		fmt.Print("Enter number of records to generate (default 100000): ")
		input, _ = reader.ReadString('\n')
		input = strings.TrimSpace(input)
		
		if input == "" {
			numRecords = 100000
		} else {
			var err error
			numRecords, err = strconv.Atoi(input)
			if err != nil || numRecords <= 0 {
				numRecords = 100000
			}
		}
		
		engine.GenerateSampleData(numRecords)
	}
	
	// Perform clustering
	start := time.Now()
	
	engine.BuildVocabulary()
	engine.KMeansCluster()
	
	duration := time.Since(start)
	fmt.Printf("Clustering completed in %v\n", duration)
	
	// Display results
	engine.PrintResults()
	engine.AnalyzeClusterQuality()
	
	// Save results
	fmt.Print("Save results to file? (y/n): ")
	save, _ := reader.ReadString('\n')
	save = strings.TrimSpace(save)
	
	if strings.ToLower(save) == "y" || strings.ToLower(save) == "yes" {
		filename := fmt.Sprintf("clustering_results_%d.json", time.Now().Unix())
		if err := engine.SaveResults(filename); err != nil {
			log.Printf("Error saving results: %v\n", err)
		} else {
			fmt.Printf("Results saved to %s\n", filename)
		}
	}
	
	fmt.Println("Clustering analysis complete!")
}