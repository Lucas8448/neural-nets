use std::io::{self, Write};

fn predict(x: f64, w: f64, b: f64) -> f64 {
    w * x + b
}

fn loss_one(x: f64, y: f64, w: f64, b: f64) -> f64 {
    let y_hat = predict(x, w, b);
    let err = y - y_hat;
    err * err
}

// Gradients for L = (y - (w*x + b))^2
fn gradients_one(x: f64, y: f64, w: f64, b: f64) -> (f64, f64) {
    let y_hat = predict(x, w, b);
    let err = y - y_hat;

    let dL_db = -2.0 * err;
    let dL_dw = -2.0 * x * err;

    (dL_dw, dL_db)
}

fn train_step_one(x: f64, y: f64, w: &mut f64, b: &mut f64, lr: f64) {
    let (dL_dw, dL_db) = gradients_one(x, y, *w, *b);
    *w -= lr * dL_dw;
    *b -= lr * dL_db;
}

fn read_f64(prompt: &str) -> f64 {
    loop {
        print!("{prompt}");
        io::stdout().flush().unwrap();

        let mut s = String::new();
        if io::stdin().read_line(&mut s).is_err() {
            println!("Failed to read input. Try again.");
            continue;
        }

        match s.trim().replace(',', ".").parse::<f64>() {
            Ok(v) if v.is_finite() => return v,
            _ => println!("Please enter a valid number (example: 2 or 2.5)."),
        }
    }
}

fn read_usize(prompt: &str) -> usize {
    loop {
        print!("{prompt}");
        io::stdout().flush().unwrap();

        let mut s = String::new();
        if io::stdin().read_line(&mut s).is_err() {
            println!("Failed to read input. Try again.");
            continue;
        }

        match s.trim().parse::<usize>() {
            Ok(v) => return v,
            _ => println!("Please enter a valid whole number (example: 1000)."),
        }
    }
}

fn main() {
    println!("Model: y_hat = w*x + b");
    println!("Loss : (y - y_hat)^2\n");

    let x = read_f64("Enter x (input): ");
    let y = read_f64("Enter y (target output): ");
    let lr = read_f64("Enter learning rate (e.g. 0.1 or 0.01): ");
    let steps = read_usize("Enter max steps (e.g. 50 or 1000): ");
    let eps = read_f64("Enter target loss epsilon (e.g. 1e-12): ");

    // Start from something simple.
    let mut w = 0.0;
    let mut b = 0.0;

    println!("\nStart: w={w:.6}, b={b:.6}");
    let mut last_loss = loss_one(x, y, w, b);
    println!("Step {:>5}: y_hat={:.6}, loss={:.6}", 0, predict(x, w, b), last_loss);

    for t in 1..=steps {
        train_step_one(x, y, &mut w, &mut b, lr);

        let y_hat = predict(x, w, b);
        let l = loss_one(x, y, w, b);

        // Print occasionally (and always near convergence)
        if t <= 10 || t % 10 == 0 || l < eps {
            println!("Step {:>5}: y_hat={:.12}, loss={:.12}   (w={:.6}, b={:.6})", t, y_hat, l, w, b);
        }

        if l < eps {
            println!("\nConverged: loss < epsilon.");
            break;
        }

        // Simple safety: if loss becomes NaN/Inf, your lr is too big.
        if !l.is_finite() {
            println!("\nLoss blew up (NaN/Inf). Try a smaller learning rate.");
            break;
        }

        // Optional: if loss is not changing much, you might be at numeric limits.
        if (last_loss - l).abs() < eps * 1e-6 && t > 20 {
            // not a perfect criterion, just a practical stopper
            println!("\nLoss improvement is tiny; stopping early.");
            break;
        }

        last_loss = l;
    }

    println!("\nFinal parameters:");
    println!("w = {:.12}", w);
    println!("b = {:.12}", b);

    let final_pred = predict(x, w, b);
    println!("\nCheck:");
    println!("For x = {x}, predicted y_hat = {final_pred}, target y = {y}");
}
