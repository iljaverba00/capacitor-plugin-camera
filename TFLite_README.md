# TensorFlow Lite Blur Detection Integration

This plugin now supports TensorFlow Lite (TFLite) models for blur detection, providing more accurate results than the traditional Laplacian algorithm. The implementation is based on the [Flutter blur detection example](https://github.com/AhmetFurkanDEMIR/Blur-image-detection-with-Flutter-and-TFLite) using MobileNetV2.

## Features

- **TFLite Model**: Uses a pre-trained MobileNetV2 model for accurate blur detection
- **Fallback Support**: Automatically falls back to Laplacian algorithm if TFLite model fails to load
- **Cross-platform**: Works on both Android and iOS
- **Performance Optimized**: Uses GPU acceleration when available
- **Easy Integration**: Drop-in replacement for existing blur detection

## Setup Instructions

### Quick Setup (Automatic)

Use the provided download script to automatically obtain and install the TFLite model:

```bash
# Make the script executable
chmod +x scripts/download_model.sh

# Run the download script
./scripts/download_model.sh
```

This script will:
- Try multiple model sources
- Validate the downloaded model
- Install it for both Android and iOS
- Provide manual instructions if automatic download fails

### Manual Setup

If the automatic download fails, you'll need to manually obtain a TFLite blur detection model:

#### Model Sources:
1. **Train your own**: Use the [Kaggle notebook](https://www.kaggle.com/code/ahmetfurkandemr/blur-detection-with-tflite)
2. **TensorFlow Hub**: Browse pre-trained models at [tfhub.dev](https://tfhub.dev/)
3. **Community models**: Search GitHub for "blur detection tflite"

#### Model Requirements:
- **Input**: Square RGB images (size auto-detected from model, commonly 224x224, 256x256, or 600x600)
- **Output**: 2 classes [sharp_probability, blur_probability] 
- **Architecture**: MobileNetV2, EfficientNet, or similar lightweight model
- **Format**: TensorFlow Lite (.tflite)
- **Size**: Typically 2-10MB

#### Manual Installation Steps:

**Android:**
1. Place your TFLite model file in `android/src/main/assets/`
2. Rename it to `blur_detection_model.tflite`

```bash
# Copy your model to the assets folder
cp your_model.tflite android/src/main/assets/blur_detection_model.tflite
```

**iOS:**
1. Place the model in the iOS plugin bundle: `ios/Plugin/blur_detection_model.tflite`
2. Or add it to your main iOS project bundle via Xcode (drag and drop, ensure "Add to target" is checked)

```bash
# Copy your model to the iOS bundle
cp your_model.tflite ios/Plugin/blur_detection_model.tflite
```

### Dependencies Installation

The TensorFlow Lite dependencies are already configured in the build files:

**Android**: TensorFlow Lite dependencies in `android/build.gradle`
```gradle
implementation 'org.tensorflow:tensorflow-lite:2.14.0'
implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'
```

**iOS**: TensorFlow Lite dependencies in `ios/Podfile`
```ruby
pod 'TensorFlowLiteSwift', '~> 2.13.0'
```

Run `pod install` in the iOS directory:
```bash
cd ios && pod install
```

## Usage

The TFLite integration is transparent to your existing code. The plugin will automatically use TFLite when available and fall back to Laplacian when not.

### JavaScript/TypeScript

```typescript
import { CameraPreview } from '@capacitor-community/camera-preview';

// Take a snapshot with blur detection
const result = await CameraPreview.takeSnapshot({
  quality: 85,
  checkBlur: true  // Enable blur detection
});

console.log('Is blurry:', result.isBlur);
// true = blurry, false = sharp

// Example usage
if (result.isBlur) {
  console.log('Image is blurry - retake recommended');
} else {
  console.log('Image is sharp - good quality');
}

// Take a photo with blur detection  
const photo = await CameraPreview.takePhoto({
  includeBase64: true
});

console.log('Is blurry:', photo.isBlur);
```

### Model Output

The TFLite implementation uses a simplified approach:

#### Pure TensorFlow Lite Classification
- Uses only the TensorFlow Lite model's output probabilities
- No hybrid scoring or image preprocessing for maximum simplicity
- Direct comparison: `probabilities[0]` (blur) vs `probabilities[1]` (sharp)
- Fallback to Laplacian algorithm only when TensorFlow Lite fails

#### Boolean Blur Detection
The system now returns a simple **boolean result**:
- **`isBlur: false`**: Image is sharp
- **`isBlur: true`**: Image is blurry

This boolean approach uses the TensorFlow Lite model's confidence comparison:
- If `probabilities[0]` (blur) > `probabilities[1]` (sharp) → `isBlur: true`
- Otherwise → `isBlur: false`

For Laplacian fallback: if score < 50 → `isBlur: true`

## Improving Accuracy

If the current accuracy isn't meeting your needs, try these approaches:

### 1. Model Selection
Look for better pre-trained models:
- **TensorFlow Hub**: Search for "blur detection" or "image quality assessment"
- **Model Zoo**: Check TensorFlow Model Garden for quality assessment models
- **Research Papers**: Recent models like BRISQUE, NIQE, or specialized blur detection CNNs

### 2. Fallback Testing
```java
// Test Laplacian fallback by temporarily removing the TFLite model
// This helps determine if the issue is with the model or the algorithm
// If Laplacian gives better results, consider training a new TFLite model
```

### 3. Custom Model Training
Train a model on your specific use case:
- Collect images similar to your camera conditions
- Include various blur types (motion, focus, gaussian)
- Use transfer learning from ImageNet models
- Consider different architectures (EfficientNet, ResNet, etc.)

## Model Training

If you want to train your own model, refer to:

1. **Kaggle Notebook**: [Blur Detection with TFLite](https://www.kaggle.com/code/ahmetfurkandemr/blur-detection-with-tflite)
2. **Flutter Example**: [GitHub Repository](https://github.com/AhmetFurkanDEMIR/Blur-image-detection-with-Flutter-and-TFLite)
3. **Dataset**: Common blur detection datasets on Kaggle
4. **CERTH Image Blur Dataset**: Professional blur detection dataset
5. **DIV2K**: High-quality images for creating blur variants

### Model Requirements

Your TFLite model should have:
- **Input**: `[1, H, W, 3]` where H=W (square input, size auto-detected)
- **Input Type**: Float32, normalized to [0, 1] (auto-normalized from [0, 255] if needed)
- **Output**: `[1, 2]` (batch, classes)
- **Output Type**: Float32 probabilities
- **Classes**: `[sharp_probability, blur_probability]`

Common input sizes: 224x224, 256x256, 512x512, 600x600

## Troubleshooting

### Common Issues

#### Tensor Size Mismatch Error
If you see errors like:
```
Error during TFLite inference: Cannot copy to a TensorFlowLite tensor (mobilenetv2_1.00_224_input) with 4320000 bytes from a Java Buffer with 602112 bytes.
```

This indicates:
- Input size mismatch between expected and actual tensor dimensions
- The model may expect a different input size than 224×224×3
- **Solution**: The latest code auto-detects the model's input size and adapts accordingly
- **Look for logs**: "Updated INPUT_SIZE to X to match model"

#### Debug Information
Enable debug logging to see detailed tensor information:

**Android**:
```bash
adb logcat -s BlurDetectionHelper
```

**Look for these log messages**:
- "Input tensor shape" - shows the model's expected input dimensions
- "Updated INPUT_SIZE to X to match model" - confirms auto-adaptation
- "TFLite Blur Detection - Blur: X, Sharp: Y, Label: Z" - shows TensorFlow Lite classification
- "Laplacian Fallback - Score: X, Label: Y" - shows fallback algorithm results

### Better Model Recommendations

If accuracy is insufficient, try these higher-quality models:

#### State-of-the-Art Models (2024)
1. **MUSIQ (Google Research)**: Multi-scale Image Quality Transformer
   - Handles native resolution images without resizing
   - Superior performance on multiple benchmarks
   - Available as TensorFlow model

2. **NTIRE 2024 Challenge Winners**: Search for models from:
   - Quality Assessment of AI-Generated Content Challenge
   - Deep Portrait Quality Assessment Challenge
   - Often achieve state-of-the-art results

3. **Training-Free Approaches**: Look for recent models using:
   - Collaborative Feature Refinement with Hausdorff distance
   - Wavelet transform-based perceptual features
   - No training data required

#### Traditional High-Quality Models
4. **BRISQUE-based models**: Search TensorFlow Hub for "BRISQUE" or "image quality"
5. **EfficientNet blur models**: Often more accurate than MobileNet
6. **ResNet50 blur detection**: Larger but more accurate models

#### Custom Training Datasets
7. **Professional Datasets**: 
   - CERTH Image Blur Dataset (professional quality)
   - LIVE Image Quality Database
   - DIV2K with synthetic blur
   - KonIQ-10k (native resolution images)
   - PaQ-2-PiQ (large-scale perceptual quality)

### Android Issues

1. **Model not found**: Ensure `blur_detection_model.tflite` is in `android/src/main/assets/`
2. **OutOfMemoryError**: Reduce image resolution or use quantized model
3. **GPU acceleration failed**: Will automatically fallback to CPU
4. **Tensor type mismatch**: Ensure model expects float32 inputs with [0,1] normalization
5. **Compilation errors**: If you see "cannot find symbol: NormalizeOp", this is expected - the code uses manual normalization for better compatibility across TensorFlow Lite versions

### iOS Issues

1. **Model not found**: Ensure model is added to iOS bundle target
2. **Metal delegate failed**: Will automatically fallback to CPU
3. **Memory issues**: Use quantized model or reduce batch size
4. **Input format issues**: Check model preprocessing requirements

### Performance Tips

1. **Use GPU acceleration**: Automatically enabled when available
2. **Model optimization**: Use quantized models for smaller size
3. **Image preprocessing**: Images are automatically resized to 224x224
4. **Fallback behavior**: Laplacian algorithm used when TFLite fails

## Logs

Monitor console logs for initialization status:

**Android**:
```
D/Camera: TFLite blur detection initialized: true
D/BlurDetectionHelper: TFLite Blur Detection - Blur: 0.998438, Sharp: 0.001562, Label: blur
D/Camera: Blur detection - Label: blur
```

**iOS**:
```
TFLite blur detection initialized: true
BlurDetectionHelper: TFLite Blur Detection - Blur: 0.998438, Sharp: 0.001562, Label: blur
```

**Fallback Example**:
```
D/BlurDetectionHelper: Error during TFLite inference: Model not loaded
D/BlurDetectionHelper: Laplacian Fallback - Score: 23.45, Label: blur
```

## Model Size and Performance

- **Model Size**: ~2-10MB (depending on quantization)
- **Inference Time**: ~50-200ms (depending on device and acceleration)
- **Memory Usage**: ~20-50MB additional during inference
- **Accuracy**: Significantly better than Laplacian for real-world blur detection

## License

The TFLite integration maintains the same license as the original plugin. Ensure your TFLite model complies with its respective license terms. 