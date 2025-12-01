# CISC473-Project-10

🚀 What This Code Does

download the clic dataset https://archive.compression.cc/2021/tasks/index.html and put it in data folder wwith data/train and data/test

Trains Compression Models:

Baseline Model: Optimizes for Mean Squared Error (MSE) to achieve high PSNR.

Perceptual Model: Optimizes for MSE + LPIPS to achieve better visual texture and realism.

FlexTok Model: Investigates a token-based architecture for high-fidelity reconstruction (experimental and in process).

Evaluates Performance:

Calculates standard metrics: PSNR, MS-SSIM, and BPP (Bits Per Pixel).

Measures perceptual quality using LPIPS.

Exports per-image results to CSV files for detailed analysis.

Visualizes Results:

Plots training vs. testing loss to monitor convergence.

📂 Project Structure & Outputs

1. Where Checkpoints are Stored

Trained model weights (.pth.tar files) are saved automatically after every epoch.

Path: experiments/<model_type>/checkpoints/

Example: experiments/baseline/checkpoints/checkpoint.pth.tar

2. Where Reconstructed Images are Stored

When you run the test script, the compressed-then-reconstructed images are saved here for visual inspection.

Path: experiments/<model_type>/reconstructions/

Example: experiments/perceptual_baseline/reconstructions/image_001.png

3. Where Metrics are Stored

The detailed performance scores for every test image are saved in a CSV file.

Path: experiments/<model_type>/test_metrics.csv

💻 How to Run

1. Train a Model (Default Settings)

To train a standard Baseline model for 100 epochs using the default configuration:

python train.py --dataset ./data/ --model_type baseline --epochs 100 --save


Defaults: Batch size = 8, Learning Rate = 1e-4, Lambda = 0.01.

2. Test a Model

To evaluate a trained model, calculate metrics, and generate reconstructed images:

python test.py --dataset ./data/ --ckpt_path ./experiments/baseline/checkpoints/checkpoint.pth.tar

3. Visualize Loss History

To plot the Training Loss vs. Testing Loss curve to check for overfitting:

python visualize_loss.py --file experiments/baseline/loss_history.csv


