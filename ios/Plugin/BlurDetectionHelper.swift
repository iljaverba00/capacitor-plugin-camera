import Foundation
import UIKit
import TensorFlowLite

/**
 * TensorFlow Lite Blur Detection Helper for iOS
 * Based on MobileNetV2 model trained for blur detection
 */
class BlurDetectionHelper {
    
    private static let TAG = "BlurDetectionHelper"
    private static let MODEL_FILENAME = "blur_detection_model.tflite"
    private static let INPUT_SIZE = 224 // MobileNetV2 standard input size
    private static let NUM_CLASSES = 2 // blur, sharp
    
    private var interpreter: Interpreter?
    private var isInitialized = false
    
    init() {
        // Initialize the helper
    }
    
    /**
     * Initialize the TFLite model
     * @return true if initialization successful
     */
    func initialize() -> Bool {
        do {
            // Load model from bundle
            guard let modelPath = Bundle.main.path(forResource: "blur_detection_model", ofType: "tflite") else {
                print("\(Self.TAG): Error - Model file not found in bundle")
                return false
            }
            
            // Configure interpreter options for better performance
            var options = Interpreter.Options()
            options.threadCount = 4 // Use multiple threads for better performance
            
            // Try to use Metal (GPU) acceleration if available
            if #available(iOS 13.0, *) {
                do {
                    let delegate = MetalDelegate()
                    options.delegates = [delegate]
                    print("\(Self.TAG): Using Metal GPU acceleration")
                } catch {
                    print("\(Self.TAG): Metal delegate not available, using CPU: \(error)")
                }
            }
            
            // Create interpreter
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            
            // Allocate memory for input and output tensors
            try interpreter?.allocateTensors()
            
            isInitialized = true
            print("\(Self.TAG): TFLite blur detection model initialized successfully")
            return true
            
        } catch {
            print("\(Self.TAG): Error initializing TFLite model: \(error)")
            isInitialized = false
            return false
        }
    }
    
    /**
     * Detect blur in image using TFLite model
     * @param image Input UIImage
     * @return Blur confidence score (scaled to match Android implementation)
     */
    func detectBlur(image: UIImage) -> Double {
        guard isInitialized, let interpreter = interpreter else {
            print("\(Self.TAG): TFLite model not initialized, falling back to Laplacian")
            let laplacianScore = calculateLaplacianBlurScore(image: image)
            let isBlur = laplacianScore < 150
            print("\(Self.TAG): Laplacian Fallback - Score: \(String(format: "%.2f", laplacianScore)), Label: \(isBlur ? "blur" : "sharp")")
            return isBlur ? 1.0 : 0.0
        }
        
        do {
            // Preprocess image for MobileNetV2
            guard let inputData = preprocessImage(image) else {
                print("\(Self.TAG): Error preprocessing image")
                return calculateLaplacianBlurScore(image: image)
            }
            
            // Copy input data to interpreter
            try interpreter.copy(inputData, toInputAt: 0)
            
            // Run inference
            try interpreter.invoke()
            
            // Get output data
            let outputTensor = try interpreter.output(at: 0)
            let outputData = outputTensor.data
            
            // Parse output probabilities (assuming float32 output)
            let probabilities = outputData.withUnsafeBytes { bytes in
                Array(bytes.bindMemory(to: Float32.self))
            }
            
            // probabilities[0] = blur probability, probabilities[1] = sharp probability
            let blurConfidence = probabilities.count > 0 ? Double(probabilities[0]) : 0.0
            let sharpConfidence = probabilities.count > 1 ? Double(probabilities[1]) : 0.0

            // Laplacian algorithm
            let laplacianScore = calculateLaplacianBlurScore(image: image)
            
            // Determine if image is blurry using TFLite confidence or Laplacian score < 50
            let isBlur = (blurConfidence > sharpConfidence && blurConfidence > 0.99) || (laplacianScore < 50 && sharpConfidence < 0.1)
            
            print("\(Self.TAG): TFLite Blur Detection - Blur: \(String(format: "%.6f", blurConfidence)), Sharp: \(String(format: "%.6f", sharpConfidence)), Label: \(isBlur ? "blur" : "sharp")")
            
            // Return 1.0 for blur, 0.0 for sharp (to maintain double return type)
            return isBlur ? 1.0 : 0.0
            
        } catch {
            print("\(Self.TAG): Error during TFLite inference: \(error)")
            // Fallback to Laplacian algorithm
            let laplacianScore = calculateLaplacianBlurScore(image: image)
            let isBlur = laplacianScore < 150
            print("\(Self.TAG): Laplacian Fallback - Score: \(String(format: "%.2f", laplacianScore)), Label: \(isBlur ? "blur" : "sharp")")
            return isBlur ? 1.0 : 0.0
        }
    }
    
    /**
     * Preprocess image for MobileNetV2 input (224x224 RGB, normalized)
     */
    private func preprocessImage(_ image: UIImage) -> Data? {
        // Resize image to 224x224
        guard let resizedImage = resizeImage(image, to: CGSize(width: Self.INPUT_SIZE, height: Self.INPUT_SIZE)) else {
            return nil
        }
        
        // Convert to pixel data
        guard let cgImage = resizedImage.cgImage else { return nil }
        
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        
        let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )
        
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Convert to float array and normalize to [0, 1]
        var normalizedPixels = [Float32]()
        normalizedPixels.reserveCapacity(width * height * 3) // RGB only
        
        for i in stride(from: 0, to: pixelData.count, by: 4) {
            let r = Float32(pixelData[i]) / 255.0
            let g = Float32(pixelData[i + 1]) / 255.0
            let b = Float32(pixelData[i + 2]) / 255.0
            normalizedPixels.append(r)
            normalizedPixels.append(g)
            normalizedPixels.append(b)
        }
        
        return Data(bytes: normalizedPixels, count: normalizedPixels.count * MemoryLayout<Float32>.size)
    }
    
    /**
     * Resize image to target size
     */
    private func resizeImage(_ image: UIImage, to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resizedImage
    }
    
    /**
     * Fallback Laplacian blur detection (from original implementation)
     */
    private func calculateLaplacianBlurScore(image: UIImage) -> Double {
        guard let cgImage = image.cgImage else { return 0.0 }
        
        let width = cgImage.width
        let height = cgImage.height
        
        // Create bitmap context to access pixel data
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return 0.0 }
        
        // Draw image into context
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let pixelData = context.data else { return 0.0 }
        let data = pixelData.bindMemory(to: UInt8.self, capacity: width * height * bytesPerPixel)
        
        // Convert to grayscale and apply Laplacian kernel
        var variance = 0.0
        var count = 0
        
        // Sample every 4th pixel for performance
        let step = 4
        for y in stride(from: step, to: height - step, by: step) {
            for x in stride(from: step, to: width - step, by: step) {
                let idx = (y * width + x) * bytesPerPixel
                
                // Convert to grayscale
                let r = Double(data[idx])
                let g = Double(data[idx + 1])
                let b = Double(data[idx + 2])
                let gray = 0.299 * r + 0.587 * g + 0.114 * b
                
                // Calculate neighbors for 3x3 Laplacian kernel
                let neighbors: [Double] = [
                    getGrayscaleValue(data: data, x: x-1, y: y-1, width: width, bytesPerPixel: bytesPerPixel),
                    getGrayscaleValue(data: data, x: x, y: y-1, width: width, bytesPerPixel: bytesPerPixel),
                    getGrayscaleValue(data: data, x: x+1, y: y-1, width: width, bytesPerPixel: bytesPerPixel),
                    getGrayscaleValue(data: data, x: x-1, y: y, width: width, bytesPerPixel: bytesPerPixel),
                    getGrayscaleValue(data: data, x: x+1, y: y, width: width, bytesPerPixel: bytesPerPixel),
                    getGrayscaleValue(data: data, x: x-1, y: y+1, width: width, bytesPerPixel: bytesPerPixel),
                    getGrayscaleValue(data: data, x: x, y: y+1, width: width, bytesPerPixel: bytesPerPixel),
                    getGrayscaleValue(data: data, x: x+1, y: y+1, width: width, bytesPerPixel: bytesPerPixel)
                ]
                
                // Apply 3x3 Laplacian kernel
                let laplacian = -neighbors[0] - neighbors[1] - neighbors[2] +
                               -neighbors[3] + 8 * gray - neighbors[4] +
                               -neighbors[5] - neighbors[6] - neighbors[7]
                
                variance += laplacian * laplacian
                count += 1
            }
        }
        
        return count > 0 ? variance / Double(count) : 0.0
    }
    
    private func getGrayscaleValue(data: UnsafePointer<UInt8>, x: Int, y: Int, width: Int, bytesPerPixel: Int) -> Double {
        let idx = (y * width + x) * bytesPerPixel
        let r = Double(data[idx])
        let g = Double(data[idx + 1])
        let b = Double(data[idx + 2])
        return 0.299 * r + 0.587 * g + 0.114 * b
    }
    


    /**
     * Check if image is blurry
     * @param image Input image
     * @return true if image is blurry, false if sharp
     */
    func isBlurry(image: UIImage) -> Bool {
        let result = detectBlur(image: image)
        return result == 1.0
    }

    /**
     * Get blur percentage (0-100%) - Deprecated, use isBlurry() instead
     * @param image Input image
     * @return Blur percentage where 0% = sharp, 100% = very blurry
     */
    @available(*, deprecated, message: "Use isBlurry() instead")
    func getBlurPercentage(image: UIImage) -> Double {
        // Convert boolean result to percentage for backward compatibility
        return isBlurry(image: image) ? 100.0 : 0.0
    }
    
    /**
     * Clean up resources
     */
    func close() {
        interpreter = nil
        isInitialized = false
        print("\(Self.TAG): TFLite blur detection model closed")
    }
    
    /**
     * Check if TFLite model is properly initialized
     */
    func getIsInitialized() -> Bool {
        return isInitialized
    }
} 