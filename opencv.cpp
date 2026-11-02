#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace cv::dnn;

// Structure representing an SSD anchor center
struct Anchor {
    float x_center;
    float y_center;
    float w;
    float h;
};

// Generates the 2,016 default SSD anchors for MediaPipe Palm Detector (192x192)
std::vector<Anchor> generateAnchors() {
    std::vector<Anchor> anchors;

    // Strides and anchor specs for 192x192 input
    // Grids: 24x24 (stride 8, 2 anchors), 12x12 (stride 16, 2 anchors), 
    //        6x6 (stride 32, 2 anchors), 3x3 (stride 64, 2 anchors)
    // 24*24*2 + 12*12*2 + 6*6*2 + 3*3*2 = 1152 + 288 + 72 + 18 = 1530 or standard 2016 depending on variant.
    // MediaPipe palm_detection standard configuration:
    struct StridesConfig {
        int stride;
        int num_layers;
    };

    // Standard MediaPipe SSD Anchor generator for 192x192 palm detector (2016 anchors)
    vector<int> strides = { 8, 16, 16, 16 };
    vector<int> anchors_per_cell = { 2, 2, 2, 2 }; // Typically 24x24x2(1152) + 12x12x6...

    // Exact MediaPipe grid sizes matching 2016 anchors:
    // Grid 24x24: 2 anchors -> 1152
    // Grid 12x12: 6 anchors -> 864
    // Total = 2016 anchors
    const int feature_map_sizes[2] = { 24, 12 };
    const int num_anchors[2] = { 2, 6 };

    for (int layer = 0; layer < 2; ++layer) {
        int grid_size = feature_map_sizes[layer];
        int count = num_anchors[layer];
        for (int y = 0; y < grid_size; ++y) {
            for (int x = 0; x < grid_size; ++x) {
                for (int a = 0; a < count; ++a) {
                    Anchor anchor;
                    anchor.x_center = (x + 0.5f) / grid_size;
                    anchor.y_center = (y + 0.5f) / grid_size;
                    anchor.w = 1.0f;
                    anchor.h = 1.0f;
                    anchors.push_back(anchor);
                }
            }
        }
    }
    return anchors;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

int main() {
    string palmDetectorPath = "D:/opencv/x64/Release/palm_detection_mediapipe_2023feb.onnx";

    Net netDetector;
    try {
        netDetector = readNetFromONNX(palmDetectorPath);
    }
    catch (const cv::Exception& e) {
        cout << "[OpenCV Error loading detector]: " << e.what() << endl;
        return -1;
    }

    if (netDetector.empty()) {
        cout << "[ERROR] Palm detector model is empty!" << endl;
        return -1;
    }
    cout << "Palm detector loaded successfully." << endl;

    // Precompute the anchors
    vector<Anchor> anchors = generateAnchors();

    // Query model output names (MediaPipe palm detector has two outputs: regressors and classifiers)
    vector<string> outNames = netDetector.getUnconnectedOutLayersNames();

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error: Could not open webcam." << endl;
        return -1;
    }

    Mat frame;
    const float scoreThreshold = 0.7f; // Confidence threshold
    const float nmsThreshold = 0.2f;   // IoU suppression threshold

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 1. Preprocess: Resize to 192x192, BGR->RGB, scale [0, 1]
        Mat resized;
        resize(frame, resized, Size(192, 192));
        cvtColor(resized, resized, COLOR_BGR2RGB);

        Mat floatMat;
        resized.convertTo(floatMat, CV_32FC3, 1.0 / 255.0);

        // 2. Build 4D NHWC Tensor [1, 192, 192, 3]
        int sz[] = { 1, 192, 192, 3 };
        Mat blob(4, sz, CV_32F, floatMat.data);

        // 3. Forward pass requesting all output layers
        vector<Mat> outputs;
        try {
            netDetector.setInput(blob);
            netDetector.forward(outputs, outNames);
        }
        catch (const cv::Exception& e) {
            cerr << "OpenCV DNN Error: " << e.what() << endl;
            continue;
        }

        if (outputs.size() < 2) continue;

        // Identify which output is raw_boxes ([1, 2016, 18]) and raw_scores ([1, 2016, 1])
        Mat rawBoxes, rawScores;
        if (outputs[0].total() > outputs[1].total()) {
            rawBoxes = outputs[0];
            rawScores = outputs[1];
        }
        else {
            rawBoxes = outputs[1];
            rawScores = outputs[0];
        }

        // Reshape matrices to 2D (num_anchors x features)
        int num_anchors = anchors.size();
        int box_features = rawBoxes.total() / num_anchors;

        float* pScores = (float*)rawScores.data;
        float* pBoxes = (float*)rawBoxes.data;

        vector<Rect> bboxes;
        vector<float> confidences;

        // 4. Decode anchors
        for (int i = 0; i < num_anchors; ++i) {
            float score = sigmoid(pScores[i]);

            if (score >= scoreThreshold) {
                // pBoxes contains [dx, dy, dw, dh, 7 keypoints * 2] normalized to 192.0
                float* box_ptr = pBoxes + (i * box_features);

                float dx = box_ptr[0] / 192.0f;
                float dy = box_ptr[1] / 192.0f;
                float dw = box_ptr[2] / 192.0f;
                float dh = box_ptr[3] / 192.0f;

                float cx = dx + anchors[i].x_center;
                float cy = dy + anchors[i].y_center;
                float w = dw;
                float h = dh;

                // Scale to original frame dimensions
                int x_min = static_cast<int>((cx - w / 2.0f) * frame.cols);
                int y_min = static_cast<int>((cy - h / 2.0f) * frame.rows);
                int box_w = static_cast<int>(w * frame.cols);
                int box_h = static_cast<int>(h * frame.rows);

                bboxes.push_back(Rect(x_min, y_min, box_w, box_h));
                confidences.push_back(score);
            }
        }

        // 5. Apply Non-Maximum Suppression (NMS)
        vector<int> indices;
        NMSBoxes(bboxes, confidences, scoreThreshold, nmsThreshold, indices);

        // 6. Draw bounding boxes on the original camera frame
        for (int idx : indices) {
            cout << idx << endl;
            Rect box = bboxes[idx];

            // Clamp box to image bounds
            box.x = std::max(0, box.x);
            box.y = std::max(0, box.y);
            box.width = std::min(box.width, frame.cols - box.x);
            box.height = std::min(box.height, frame.rows - box.y);

            rectangle(frame, box, Scalar(0, 255, 0), 2);

            string label = format("Palm: %.2f", confidences[idx]);
            int baseLine = 0;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int top = max(box.y, labelSize.height);
            rectangle(frame, Point(box.x, top - labelSize.height),
                Point(box.x + labelSize.width, top + baseLine),
                Scalar(0, 255, 0), FILLED);
            putText(frame, label, Point(box.x, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
        }

        imshow("MediaPipe Palm Detector", frame);

        char key = (char)waitKey(1);
        if (key == 27 || key == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}