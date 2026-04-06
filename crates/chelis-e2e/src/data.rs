use chelis_ir::eval::TensorValue;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// A batch of (input, label) tensor pairs.
pub type BatchVec = Vec<(TensorValue, TensorValue)>;

/// Generate synthetic random data for sanity checking
pub fn synthetic_data(
    n_samples: usize,
    batch_size: usize,
    seed: u64,
) -> Vec<(TensorValue, TensorValue)> {
    let mut state = seed;
    let mut next_f64 = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64) / (1u64 << 31) as f64
    };

    let mut batches = Vec::new();
    let mut i = 0;
    while i < n_samples {
        let bs = batch_size.min(n_samples - i);
        let x_data: Vec<f64> = (0..bs * 784).map(|_| next_f64()).collect();
        let x = TensorValue::from_vec(vec![bs, 784], x_data);

        let mut y_data = vec![0.0; bs * 10];
        for b in 0..bs {
            let label = ((next_f64() * 10.0) as usize).min(9);
            y_data[b * 10 + label] = 1.0;
        }
        let y = TensorValue::from_vec(vec![bs, 10], y_data);

        batches.push((x, y));
        i += bs;
    }
    batches
}

/// Load MNIST IDX files from a directory
pub fn load_mnist(dir: &Path) -> Result<(BatchVec, BatchVec), String> {
    let train_images = load_idx_images(&dir.join("train-images-idx3-ubyte"))?;
    let train_labels = load_idx_labels(&dir.join("train-labels-idx1-ubyte"))?;
    let test_images = load_idx_images(&dir.join("t10k-images-idx3-ubyte"))?;
    let test_labels = load_idx_labels(&dir.join("t10k-labels-idx1-ubyte"))?;

    let train = batch_data(&train_images, &train_labels, 32);
    let test = batch_data(&test_images, &test_labels, 32);

    Ok((train, test))
}

pub fn default_mnist_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("MNIST_DIR") {
        return PathBuf::from(dir);
    }

    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/mnist")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from("data/mnist"))
}

fn load_idx_images(path: &Path) -> Result<Vec<Vec<f64>>, String> {
    let mut file = File::open(path).map_err(|e| format!("can't open {}: {e}", path.display()))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .map_err(|e| format!("read error: {e}"))?;

    let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if magic != 2051 {
        return Err(format!("bad magic: {magic}"));
    }
    let n = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let rows = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
    let cols = u32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]) as usize;

    let mut images = Vec::with_capacity(n);
    for i in 0..n {
        let start = 16 + i * rows * cols;
        let pixels: Vec<f64> = buf[start..start + rows * cols]
            .iter()
            .map(|&b| b as f64 / 255.0)
            .collect();
        images.push(pixels);
    }
    Ok(images)
}

fn load_idx_labels(path: &Path) -> Result<Vec<usize>, String> {
    let mut file = File::open(path).map_err(|e| format!("can't open {}: {e}", path.display()))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .map_err(|e| format!("read error: {e}"))?;

    let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if magic != 2049 {
        return Err(format!("bad magic: {magic}"));
    }
    let n = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;

    Ok(buf[8..8 + n].iter().map(|&b| b as usize).collect())
}

fn batch_data(
    images: &[Vec<f64>],
    labels: &[usize],
    batch_size: usize,
) -> Vec<(TensorValue, TensorValue)> {
    let mut batches = Vec::new();
    let mut i = 0;
    while i + batch_size <= images.len() {
        let x_data: Vec<f64> = images[i..i + batch_size]
            .iter()
            .flat_map(|img| img.iter().copied())
            .collect();
        let x = TensorValue::from_vec(vec![batch_size, 784], x_data);

        let mut y_data = vec![0.0; batch_size * 10];
        for (b, &label) in labels[i..i + batch_size].iter().enumerate() {
            y_data[b * 10 + label] = 1.0;
        }
        let y = TensorValue::from_vec(vec![batch_size, 10], y_data);

        batches.push((x, y));
        i += batch_size;
    }
    batches
}
