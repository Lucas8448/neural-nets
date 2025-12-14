use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// WORD-LEVEL BIGRAM MODEL
// ============================================================================
//
// A bigram model predicts the next token based ONLY on the previous token.
// It's the simplest form of a language model.
//
// KEY CONCEPTS:
//
// 1. TOKENIZATION: We split text into words (tokens). Each unique word gets
//    an integer ID for efficient storage and lookup.
//
// 2. BIGRAM COUNTS: We count how often each word follows another word.
//    For example, in "the cat sat on the mat":
//      - "the" is followed by "cat" once and "mat" once
//      - "cat" is followed by "sat" once
//      - etc.
//
// 3. PROBABILITY: To predict the next word after "the", we look at all words
//    that ever followed "the" and sample proportionally to their counts.
//    P(next | prev) = count(prev, next) / sum of all counts for prev
//
// 4. GENERATION: Start with a seed word, repeatedly sample the next word
//    using the learned probabilities.
//
// LIMITATIONS:
// - Only considers ONE previous word (no long-range context)
// - Can't handle words it hasn't seen before (out-of-vocabulary)
// - Generated text often lacks coherence beyond 2-word phrases
//
// ============================================================================

fn main() {
    // Load training text
    let text = std::fs::read_to_string("src/bin/training.txt")
        .expect("Failed to read training.txt");

    // Build vocabulary and train the word-level bigram model
    let (vocab, inv_vocab, counts) = train_word_bigrams(&text);

    println!("Vocabulary size: {} words", vocab.len());
    println!();

    // Generate text
    let seed = time_seed();
    let mut rng = Rng::new(seed);

    // Pick a random starting word from vocabulary
    let start_word = "The";
    let start_id = vocab.get(start_word).copied().unwrap_or(0);

    let generated_ids = generate_words(&counts, start_id, 5000, &mut rng);
    let generated_text: Vec<&str> = generated_ids
        .iter()
        .map(|&id| inv_vocab[id].as_str())
        .collect();

    println!("--- Generated (Word-Level Bigram) ---");
    println!("{}", generated_text.join(" "));
}

// ============================================================================
// VOCABULARY BUILDING
// ============================================================================
//
// We need to map words <-> integers because:
// 1. Integers are faster to compare and use as array indices
// 2. We can use a 2D array/matrix for efficient count storage
//
// vocab: word -> id (for encoding)
// inv_vocab: id -> word (for decoding back to text)

fn build_vocabulary(text: &str) -> (HashMap<String, usize>, Vec<String>) {
    let mut vocab: HashMap<String, usize> = HashMap::new();
    let mut inv_vocab: Vec<String> = Vec::new();

    for word in text.split_whitespace() {
        if !vocab.contains_key(word) {
            let id = inv_vocab.len();
            vocab.insert(word.to_string(), id);
            inv_vocab.push(word.to_string());
        }
    }

    (vocab, inv_vocab)
}

// ============================================================================
// TRAINING: Count bigrams
// ============================================================================
//
// For each consecutive pair of words (w1, w2), increment counts[w1_id][w2_id].
// This builds a sparse matrix of transition counts.
//
// Example: "I like cats I like dogs"
//   Words: ["I", "like", "cats", "I", "like", "dogs"]
//   IDs:   [0, 1, 2, 0, 1, 3]
//   Bigrams: (0,1), (1,2), (2,0), (0,1), (1,3)
//   
//   counts[0][1] = 2  (I -> like appears twice)
//   counts[1][2] = 1  (like -> cats)
//   counts[1][3] = 1  (like -> dogs)
//   counts[2][0] = 1  (cats -> I)

fn train_word_bigrams(text: &str) -> (HashMap<String, usize>, Vec<String>, HashMap<usize, HashMap<usize, u32>>) {
    let (vocab, inv_vocab) = build_vocabulary(text);

    // Sparse storage: HashMap<prev_id, HashMap<next_id, count>>
    // More memory-efficient than a dense VxV matrix for large vocabularies
    let mut counts: HashMap<usize, HashMap<usize, u32>> = HashMap::new();

    let words: Vec<&str> = text.split_whitespace().collect();

    if words.len() < 2 {
        return (vocab, inv_vocab, counts);
    }

    // Count all bigrams
    for i in 0..(words.len() - 1) {
        let prev_id = vocab[words[i]];
        let next_id = vocab[words[i + 1]];

        *counts
            .entry(prev_id)
            .or_insert_with(HashMap::new)
            .entry(next_id)
            .or_insert(0) += 1;
    }

    (vocab, inv_vocab, counts)
}

// ============================================================================
// GENERATION: Sample from learned distribution
// ============================================================================
//
// Given the current word's ID, look up which words followed it during training.
// Sample the next word proportionally to the counts (weighted random selection).
//
// This is essentially sampling from P(next | prev) = count(prev,next) / total

fn generate_words(
    counts: &HashMap<usize, HashMap<usize, u32>>,
    start_id: usize,
    length: usize,
    rng: &mut Rng,
) -> Vec<usize> {
    let mut out = Vec::with_capacity(length + 1);
    out.push(start_id);

    let mut prev_id = start_id;

    for _ in 0..length {
        let next_id = sample_next_word(counts, prev_id, rng);
        out.push(next_id);
        prev_id = next_id;
    }

    out
}

fn sample_next_word(
    counts: &HashMap<usize, HashMap<usize, u32>>,
    prev_id: usize,
    rng: &mut Rng,
) -> usize {
    // Get the distribution of words that follow prev_id
    let Some(next_counts) = counts.get(&prev_id) else {
        // Never seen this word - fall back to word 0
        return 0;
    };

    // Sum all counts to get the total "probability mass"
    let total: u64 = next_counts.values().map(|&c| c as u64).sum();

    if total == 0 {
        return 0;
    }

    // Pick a random point in [0, total)
    let mut r = rng.next_u64() % total;

    // Walk through the distribution until we land on a word
    // This is "roulette wheel" selection - words with higher counts
    // occupy more of the wheel and are more likely to be selected
    for (&next_id, &count) in next_counts {
        let w = count as u64;
        if r < w {
            return next_id;
        }
        r -= w;
    }

    // Fallback (shouldn't happen if counts are correct)
    0
}

// ============================================================================
// UTILITIES
// ============================================================================

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        let s = if seed == 0 { 0xdead_beef_cafe_f00d } else { seed };
        Self { state: s }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64: simple and fast PRNG
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

fn time_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}
