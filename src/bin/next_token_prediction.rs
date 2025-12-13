use std::time::{SystemTime, UNIX_EPOCH};

const N: usize = 256; // byte alphabet size (0..=255)

fn main() {
    // Training text from training.txt
    let text = std::fs::read_to_string("src/bin/training.txt").expect("Failed to read training.txt");

    // 1) Train bigram counts: counts[prev][next] += 1
    let counts = train_bigrams(text.as_bytes());

    // 2) Generate text
    let seed = time_seed();
    let mut rng = Rng::new(seed);

    let start = b'R';
    let generated = generate(&counts, start, 400, &mut rng);

    println!("--- Generated ---");
    println!("{}", String::from_utf8_lossy(&generated));
}

// ---------------------------
// Training
// ---------------------------

fn train_bigrams(data: &[u8]) -> [[u32; N]; N] {
    let mut counts = [[0u32; N]; N];

    if data.len() < 2 {
        return counts;
    }

    for i in 0..(data.len() - 1) {
        let prev = data[i] as usize;
        let next = data[i + 1] as usize;
        counts[prev][next] += 1;
    }

    counts
}

// ---------------------------
// Generation
// ---------------------------

fn generate(counts: &[[u32; N]; N], start: u8, length: usize, rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::with_capacity(length + 1);
    out.push(start);

    let mut prev = start;

    for _ in 0..length {
        let next = sample_next(counts, prev, rng);
        out.push(next);
        prev = next;
    }

    out
}

fn sample_next(counts: &[[u32; N]; N], prev: u8, rng: &mut Rng) -> u8 {
    let row = &counts[prev as usize];

    // Only sum counts that actually appeared (no +1 smoothing across all 256 bytes), known as Laplacian smoothing
    let total: u64 = row.iter().map(|&c| c as u64).sum();

    // If we've never seen this character before, fall back to a space or newline
    if total == 0 {
        return b' ';
    }

    // Pick a random number in [0, total)
    let mut r = rng.next_u64() % total;

    // Walk through weights until we land somewhere
    for (i, &c) in row.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let w = c as u64;
        if r < w {
            return i as u8;
        }
        r -= w;
    }

    // Fallback (shouldn't happen)
    b' '
}

// Tiny RNG (so we don't need rand crate)

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        // Avoid seed=0 getting stuck in some RNGs
        let s = if seed == 0 { 0xdead_beef_cafe_f00d } else { seed };
        Self { state: s }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64: small, simple, decent for demos
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
