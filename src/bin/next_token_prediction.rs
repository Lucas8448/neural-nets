use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    // Load training text
    let text =
        std::fs::read_to_string("src/bin/training.txt").expect("Failed to read training.txt");

    let seed = time_seed();
    
    // ========================================
    // Method 1: Pure character-level bigrams
    // ========================================
    println!("=== Method 1: Character-level Bigrams ===");
    println!("Text length: {} bytes/characters", text.len());
    
    let char_tokens: Vec<usize> = text.as_bytes().iter().map(|&b| b as usize).collect();
    let char_counts = train_token_bigrams(&char_tokens);
    
    let mut rng1 = Rng::new(seed);
    let start_char = b'T' as usize;
    let char_generated = generate_bigram_tokens(&char_counts, start_char, 500, &mut rng1);
    let char_text: String = char_generated.iter().map(|&id| id as u8 as char).collect();
    
    println!("--- Generated (Character Bigrams) ---");
    println!("{}", char_text);
    println!();

    // ========================================
    // Method 2: BPE tokenizer + TRIGRAMS
    // ========================================
    println!("=== Method 2: BPE Tokenizer + Trigrams ===");
    println!("Training BPE tokenizer...");

    // With trigrams, we can use more merges since we have
    // more context to disambiguate
    let num_merges = 300;
    let tokenizer = BpeTokenizer::train(&text, num_merges);

    println!("Vocabulary size: {} tokens", tokenizer.vocab_size());
    println!();

    // Show some example tokens
    println!("Sample learned tokens:");
    for (i, token) in tokenizer.inv_vocab.iter().enumerate().skip(256).take(20) {
        println!("  Token {}: {:?}", i, String::from_utf8_lossy(token));
    }
    println!();

    // Tokenize the training text
    let token_ids = tokenizer.encode(&text);
    println!(
        "Compression: {} bytes -> {} tokens ({:.1}x)",
        text.len(),
        token_ids.len(),
        text.len() as f64 / token_ids.len() as f64
    );
    println!();

    // Train TRIGRAM model on tokens (looks at previous 2 tokens)
    let trigram_counts = train_token_trigrams(&token_ids);
    println!("Trigram contexts learned: {}", trigram_counts.len());
    println!();

    // Generate text using trigrams
    let mut rng2 = Rng::new(seed);

    // Start with "The " tokens
    let start_text = "The ";
    let start_ids = tokenizer.encode(start_text);
    let (ctx1, ctx2) = if start_ids.len() >= 2 {
        (start_ids[start_ids.len() - 2], start_ids[start_ids.len() - 1])
    } else if start_ids.len() == 1 {
        (b' ' as usize, start_ids[0])
    } else {
        (b' ' as usize, b'T' as usize)
    };

    let generated_ids = generate_trigram_tokens(&trigram_counts, ctx1, ctx2, 200, &mut rng2);
    let generated_text = tokenizer.decode(&generated_ids);

    println!("--- Generated (BPE Tokenizer + Trigram) ---");
    println!("{}", generated_text);
}

// ============================================================================
// BPE TOKENIZER
// ============================================================================

struct BpeTokenizer {
    // Merge rules: (token1, token2) -> merged_token_id
    // Applied in order during encoding
    merges: Vec<((usize, usize), usize)>,

    // Vocabulary: token_id -> byte sequence
    inv_vocab: Vec<Vec<u8>>,

    // Reverse lookup: byte sequence -> token_id (for encoding)
    vocab: HashMap<Vec<u8>, usize>,
}

impl BpeTokenizer {
    /// Train a BPE tokenizer on the given text
    fn train(text: &str, num_merges: usize) -> Self {
        let bytes = text.as_bytes();

        // Initialize vocabulary with all 256 possible bytes
        let mut inv_vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let mut vocab: HashMap<Vec<u8>, usize> =
            inv_vocab.iter().cloned().enumerate().map(|(i, v)| (v, i)).collect();

        // Current tokenization of the text (start as individual bytes)
        let mut tokens: Vec<usize> = bytes.iter().map(|&b| b as usize).collect();

        let mut merges: Vec<((usize, usize), usize)> = Vec::with_capacity(num_merges);

        for i in 0..num_merges {
            if tokens.len() < 2 {
                break;
            }

            // Count all adjacent pairs
            let pair_counts = count_pairs(&tokens);

            // Find the most frequent pair
            let Some((best_pair, count)) = pair_counts.iter().max_by_key(|(_, c)| *c) else {
                break;
            };

            if *count < 2 {
                // No pair appears more than once, stop
                break;
            }

            // Create new token by concatenating the pair
            let (t1, t2) = *best_pair;
            let mut new_token = inv_vocab[t1].clone();
            new_token.extend(&inv_vocab[t2]);

            let new_id = inv_vocab.len();
            inv_vocab.push(new_token.clone());
            vocab.insert(new_token, new_id);
            merges.push((*best_pair, new_id));

            // Replace all occurrences of the pair with the new token
            tokens = merge_pair(&tokens, *best_pair, new_id);

            if (i + 1) % 100 == 0 {
                println!(
                    "  Merge {}/{}: {:?} + {:?} -> token {} (count: {})",
                    i + 1,
                    num_merges,
                    String::from_utf8_lossy(&inv_vocab[t1]),
                    String::from_utf8_lossy(&inv_vocab[t2]),
                    new_id,
                    count
                );
            }
        }

        Self { merges, inv_vocab, vocab }
    }

    fn vocab_size(&self) -> usize {
        self.inv_vocab.len()
    }

    /// Encode text into token IDs
    fn encode(&self, text: &str) -> Vec<usize> {
        // Start with bytes
        let mut tokens: Vec<usize> = text.as_bytes().iter().map(|&b| b as usize).collect();

        // Apply merges in order (greedy)
        for &(pair, new_id) in &self.merges {
            tokens = merge_pair(&tokens, pair, new_id);
        }

        tokens
    }

    /// Decode token IDs back to text
    fn decode(&self, tokens: &[usize]) -> String {
        let bytes: Vec<u8> = tokens
            .iter()
            .filter_map(|&id| self.inv_vocab.get(id))
            .flatten()
            .copied()
            .collect();

        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Count all adjacent pairs in the token sequence
fn count_pairs(tokens: &[usize]) -> HashMap<(usize, usize), u32> {
    let mut counts = HashMap::new();

    for window in tokens.windows(2) {
        let pair = (window[0], window[1]);
        *counts.entry(pair).or_insert(0) += 1;
    }

    counts
}

/// Replace all occurrences of `pair` with `new_id`
fn merge_pair(tokens: &[usize], pair: (usize, usize), new_id: usize) -> Vec<usize> {
    let mut result = Vec::with_capacity(tokens.len());
    let mut i = 0;

    while i < tokens.len() {
        if i + 1 < tokens.len() && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
            result.push(new_id);
            i += 2;
        } else {
            result.push(tokens[i]);
            i += 1;
        }
    }

    result
}

// ============================================================================
// BIGRAM MODEL ON TOKENS
// ============================================================================

fn train_token_bigrams(tokens: &[usize]) -> HashMap<usize, HashMap<usize, u32>> {
    let mut counts: HashMap<usize, HashMap<usize, u32>> = HashMap::new();

    if tokens.len() < 2 {
        return counts;
    }

    for window in tokens.windows(2) {
        let prev_id = window[0];
        let next_id = window[1];

        *counts
            .entry(prev_id)
            .or_default()
            .entry(next_id)
            .or_insert(0) += 1;
    }

    counts
}

fn generate_bigram_tokens(
    counts: &HashMap<usize, HashMap<usize, u32>>,
    start_id: usize,
    length: usize,
    rng: &mut Rng,
) -> Vec<usize> {
    let mut out = Vec::with_capacity(length + 1);
    out.push(start_id);

    let mut prev_id = start_id;

    for _ in 0..length {
        let next_id = sample_bigram_token(counts, prev_id, rng);
        out.push(next_id);
        prev_id = next_id;
    }

    out
}

fn sample_bigram_token(
    counts: &HashMap<usize, HashMap<usize, u32>>,
    prev_id: usize,
    rng: &mut Rng,
) -> usize {
    let Some(next_counts) = counts.get(&prev_id) else {
        return b' ' as usize;
    };

    let total: u64 = next_counts.values().map(|&c| c as u64).sum();
    if total == 0 {
        return b' ' as usize;
    }

    let mut r = rng.next_u64() % total;
    for (&next_id, &count) in next_counts {
        if r < count as u64 {
            return next_id;
        }
        r -= count as u64;
    }
    b' ' as usize
}

// ============================================================================
// TRIGRAM MODEL ON TOKENS
// ============================================================================

/// Train trigram model: (token_i-2, token_i-1) -> token_i
fn train_token_trigrams(tokens: &[usize]) -> HashMap<(usize, usize), HashMap<usize, u32>> {
    let mut counts: HashMap<(usize, usize), HashMap<usize, u32>> = HashMap::new();

    if tokens.len() < 3 {
        return counts;
    }

    for window in tokens.windows(3) {
        let ctx = (window[0], window[1]);
        let next_id = window[2];

        *counts
            .entry(ctx)
            .or_default()
            .entry(next_id)
            .or_insert(0) += 1;
    }

    counts
}

fn generate_trigram_tokens(
    counts: &HashMap<(usize, usize), HashMap<usize, u32>>,
    ctx1: usize,
    ctx2: usize,
    length: usize,
    rng: &mut Rng,
) -> Vec<usize> {
    let mut out = Vec::with_capacity(length + 2);
    out.push(ctx1);
    out.push(ctx2);

    let mut prev1 = ctx1;
    let mut prev2 = ctx2;

    for _ in 0..length {
        let next_id = sample_trigram_token(counts, prev1, prev2, rng);
        out.push(next_id);
        prev1 = prev2;
        prev2 = next_id;
    }

    out
}

fn sample_trigram_token(
    counts: &HashMap<(usize, usize), HashMap<usize, u32>>,
    prev1: usize,
    prev2: usize,
    rng: &mut Rng,
) -> usize {
    let ctx = (prev1, prev2);
    
    let Some(next_counts) = counts.get(&ctx) else {
        // Fallback: return space
        return b' ' as usize;
    };

    let total: u64 = next_counts.values().map(|&c| c as u64).sum();
    if total == 0 {
        return b' ' as usize;
    }

    let mut r = rng.next_u64() % total;
    for (&next_id, &count) in next_counts {
        if r < count as u64 {
            return next_id;
        }
        r -= count as u64;
    }
    b' ' as usize
}

// ============================================================================
// UTILITIES
// ============================================================================

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        let s = if seed == 0 {
            0xdead_beef_cafe_f00d
        } else {
            seed
        };
        Self { state: s }
    }

    fn next_u64(&mut self) -> u64 {
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
