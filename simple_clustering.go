package main

import (
	"fmt"
	"math"
	"math/rand"
	"regexp"
	"strings"
	"time"
)

// Record holds ID and description
type Record struct {
	ID   string
	Text string
}

// Cluster represents a group of records
type Cluster struct {
	ID      int
	Records []Record
}

// Simple clustering engine
type SimpleClustering struct {
	Records []Record
	Words   []string      // vocabulary
	Vectors [][]float64   // TF-IDF vectors
	K       int           // number of clusters
}

func main() {
	// Create sample data
	records := generateSampleData(1000) // Start with 1000 for demo
	
	// Create clustering engine
	clustering := &SimpleClustering{
		Records: records,
		K:       5, // 5 clusters
	}
	
	// Run clustering
	fmt.Println("Starting clustering...")
	clusters := clustering.cluster()
	
	// Print results
	printResults(clusters)
}

// Generate simple sample data
func generateSampleData(count int) []Record {
	topics := []string{
		"computer software programming",
		"medical health doctor",
		"education school student",
		"business finance money",
		"sports game team",
	}
	
	records := make([]Record, count)
	rand.Seed(time.Now().UnixNano())
	
	for i := 0; i < count; i++ {
		topic := topics[rand.Intn(len(topics))]
		records[i] = Record{
			ID:   fmt.Sprintf("R%d", i+1),
			Text: topic + " example description",
		}
	}
	
	return records
}

// Main clustering function
func (sc *SimpleClustering) cluster() []Cluster {
	// Step 1: Build vocabulary
	sc.buildVocabulary()
	
	// Step 2: Convert text to vectors
	sc.vectorize()
	
	// Step 3: Run K-means
	return sc.kmeans()
}

// Build simple vocabulary from all text
func (sc *SimpleClustering) buildVocabulary() {
	wordCount := make(map[string]int)
	
	for _, record := range sc.Records {
		words := cleanText(record.Text)
		for _, word := range words {
			wordCount[word]++
		}
	}
	
	// Keep words that appear at least twice
	for word, count := range wordCount {
		if count >= 2 {
			sc.Words = append(sc.Words, word)
		}
	}
	
	fmt.Printf("Vocabulary size: %d words\n", len(sc.Words))
}

// Clean and split text into words
func cleanText(text string) []string {
	// Convert to lowercase
	text = strings.ToLower(text)
	
	// Remove non-letters
	reg := regexp.MustCompile(`[^a-z\s]`)
	text = reg.ReplaceAllString(text, "")
	
	// Split into words
	words := strings.Fields(text)
	
	// Remove short words
	var filtered []string
	for _, word := range words {
		if len(word) > 2 {
			filtered = append(filtered, word)
		}
	}
	
	return filtered
}

// Convert all text to TF-IDF vectors
func (sc *SimpleClustering) vectorize() {
	sc.Vectors = make([][]float64, len(sc.Records))
	
	for i, record := range sc.Records {
		sc.Vectors[i] = sc.textToVector(record.Text)
	}
	
	fmt.Printf("Vectorized %d records\n", len(sc.Vectors))
}

// Convert single text to TF-IDF vector
func (sc *SimpleClustering) textToVector(text string) []float64 {
	words := cleanText(text)
	vector := make([]float64, len(sc.Words))
	
	// Count word frequencies
	wordCount := make(map[string]int)
	for _, word := range words {
		wordCount[word]++
	}
	
	// Calculate TF for each vocabulary word
	for i, vocabWord := range sc.Words {
		if count, exists := wordCount[vocabWord]; exists {
			tf := float64(count) / float64(len(words))
			vector[i] = tf // Simplified: just use TF, skip IDF
		}
	}
	
	return vector
}

// Simple K-means clustering
func (sc *SimpleClustering) kmeans() []Cluster {
	// Initialize random centroids
	centroids := make([][]float64, sc.K)
	for i := range centroids {
		centroids[i] = make([]float64, len(sc.Words))
		for j := range centroids[i] {
			centroids[i][j] = rand.Float64()
		}
	}
	
	// Run iterations
	for iter := 0; iter < 20; iter++ { // Fixed 20 iterations
		// Assign points to clusters
		assignments := make([]int, len(sc.Vectors))
		for i, vector := range sc.Vectors {
			bestCluster := 0
			minDistance := distance(vector, centroids[0])
			
			for j := 1; j < sc.K; j++ {
				d := distance(vector, centroids[j])
				if d < minDistance {
					minDistance = d
					bestCluster = j
				}
			}
			assignments[i] = bestCluster
		}
		
		// Update centroids
		newCentroids := make([][]float64, sc.K)
		counts := make([]int, sc.K)
		
		for i := range newCentroids {
			newCentroids[i] = make([]float64, len(sc.Words))
		}
		
		for i, vector := range sc.Vectors {
			cluster := assignments[i]
			counts[cluster]++
			for j, val := range vector {
				newCentroids[cluster][j] += val
			}
		}
		
		// Average the centroids
		for i := range newCentroids {
			if counts[i] > 0 {
				for j := range newCentroids[i] {
					newCentroids[i][j] /= float64(counts[i])
				}
			}
		}
		
		centroids = newCentroids
	}
	
	// Create final clusters
	clusters := make([]Cluster, sc.K)
	for i := range clusters {
		clusters[i] = Cluster{ID: i, Records: []Record{}}
	}
	
	// Final assignment
	for i, vector := range sc.Vectors {
		bestCluster := 0
		minDistance := distance(vector, centroids[0])
		
		for j := 1; j < sc.K; j++ {
			d := distance(vector, centroids[j])
			if d < minDistance {
				minDistance = d
				bestCluster = j
			}
		}
		
		clusters[bestCluster].Records = append(clusters[bestCluster].Records, sc.Records[i])
	}
	
	return clusters
}

// Calculate Euclidean distance between two vectors
func distance(v1, v2 []float64) float64 {
	sum := 0.0
	for i := range v1 {
		diff := v1[i] - v2[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// Print clustering results
func printResults(clusters []Cluster) {
	fmt.Println("\n=== CLUSTERING RESULTS ===")
	
	for _, cluster := range clusters {
		fmt.Printf("\nCluster %d (%d records):\n", cluster.ID, len(cluster.Records))
		
		// Show first 3 examples
		max := 3
		if len(cluster.Records) < max {
			max = len(cluster.Records)
		}
		
		for i := 0; i < max; i++ {
			fmt.Printf("  %s: %s\n", cluster.Records[i].ID, cluster.Records[i].Text)
		}
		
		if len(cluster.Records) > max {
			fmt.Printf("  ... and %d more\n", len(cluster.Records)-max)
		}
	}
}