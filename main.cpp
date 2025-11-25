#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

#include <opencv2/opencv.hpp>

// NCNN
#include "net.h"
#include "cpu.h"

// ----------------------------
// CONFIG
// ----------------------------
#define DEVICE "/dev/video0"
#define WIDTH 1280
#define HEIGHT 720
#define CROP_SIZE 640

struct Buffer {
    void* start;
    size_t length;
};

std::atomic<bool> running(true);
std::mutex frameMutex;
cv::Mat latestFrame;

// ======================================================
// CAPTURE THREAD  → MJPEG V4L2 → crop 640×640
// ======================================================
void captureThread() {
    int fd = open(DEVICE, O_RDWR);
    if (fd < 0) {
        std::cerr << "Error opening V4L2 device\n";
        running = false;
        return;
    }

    // Configure to MJPEG
    v4l2_format fmt {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width  = WIDTH;
    fmt.fmt.pix.height = HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        std::cerr << "Error setting format\n";
        running = false;
        return;
    }

    // Request buffers
    v4l2_requestbuffers req {};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
        std::cerr << "Error requesting buffers\n";
        running = false;
        return;
    }

    Buffer* buffers = new Buffer[req.count];

    // Map buffers
    for (unsigned i = 0; i < req.count; i++) {
        v4l2_buffer buf {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        ioctl(fd, VIDIOC_QUERYBUF, &buf);

        buffers[i].length = buf.length;
        buffers[i].start = mmap(NULL, buf.length,
                                PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, buf.m.offset);

        ioctl(fd, VIDIOC_QBUF, &buf);
    }

    // Start stream
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd, VIDIOC_STREAMON, &type);

    while (running) {
        v4l2_buffer buf {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        ioctl(fd, VIDIOC_DQBUF, &buf);

        // Copy JPEG bytes
        std::vector<uchar> jpegData(buf.bytesused);
        memcpy(jpegData.data(), buffers[buf.index].start, buf.bytesused);

        // Decode MJPEG
        cv::Mat frame = cv::imdecode(jpegData, cv::IMREAD_COLOR);

        if (!frame.empty()) {
            // CENTER CROP 640×640
            int startX = (frame.cols - CROP_SIZE) / 2;
            int startY = (frame.rows - CROP_SIZE) / 2;

            startX = std::max(0, startX);
            startY = std::max(0, startY);

            cv::Rect roi(startX, startY, CROP_SIZE, CROP_SIZE);
            cv::Mat square = frame(roi).clone();

            // Store cropped frame
            std::lock_guard<std::mutex> lock(frameMutex);
            latestFrame = square;
        }

        ioctl(fd, VIDIOC_QBUF, &buf);
    }

    // Stop streaming
    ioctl(fd, VIDIOC_STREAMOFF, &type);

    for (unsigned i = 0; i < req.count; i++) {
        munmap(buffers[i].start, buffers[i].length);
    }

    delete[] buffers;
    close(fd);
}


// ======================================================
// MAIN THREAD — display + YOLO forward pass (CPU)
// ======================================================

ncnn::Net yolo;

int main() {
    // NCNN CPU tuning (optional, but good on RPi5)
    ncnn::set_cpu_powersave(0);       // 0: all cores
    ncnn::set_omp_num_threads(4);     // if NCNN built with OpenMP

    // YOLO options
    yolo.opt.num_threads = 4;
    yolo.opt.use_packing_layout = true;
    yolo.opt.use_sgemm_convolution = true;
    yolo.opt.use_winograd_convolution = true;
    yolo.opt.use_fp16_packed = true;
    yolo.opt.use_fp16_storage = true;
    yolo.opt.use_fp16_arithmetic = false;  // safe default

    // Load YOLOv11n NCNN model (exported as best.param/best.bin)
    if (yolo.load_param("best.param")) {
        std::cerr << "Error loading param file\n";
        return -1;
    }
    if (yolo.load_model("best.bin")) {
        std::cerr << "Error loading bin file\n";
        return -1;
    }

    std::cout << "YOLO model loaded successfully.\n";

    std::thread camThread(captureThread);

    double fps = 0;
    int count = 0;
    double tStart = cv::getTickCount();

    // For debug: print output shape about once per second
    int printed = 0;

    while (running) {
        cv::Mat frame;

        // Get CROPPED frame (640x640)
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            if (!latestFrame.empty())
                frame = latestFrame.clone();
        }

        if (!frame.empty()) {
            // Convert CROPPED frame → NCNN Mat
            ncnn::Mat in = ncnn::Mat::from_pixels(
                frame.data,
                ncnn::Mat::PIXEL_BGR,
                frame.cols,   // 640
                frame.rows    // 640
            );

            // Normalize 0–1 (same as Ultralytics)
            const float norm_vals[3] = {
                1.0f / 255.0f,
                1.0f / 255.0f,
                1.0f / 255.0f
            };
            in.substract_mean_normalize(nullptr, norm_vals);

            // -------------------------------
            // YOLO FORWARD PASS (CPU)
            // -------------------------------
            ncnn::Extractor ex = yolo.create_extractor();
            
            // For Ultralytics NCNN export, input is usually "in0"
            ex.input("in0", in);

            ncnn::Mat out;
            // And output often "out0" (you saw that in Python logs)
            ex.extract("out0", out);

            // Print shape once per ~second
            if (printed == 0) {
                std::cout << "YOLO out: w=" << out.w
                          << " h=" << out.h
                          << " c=" << out.c << std::endl;
                printed = 1;
            }

            // FPS calculation
            count++;
            double tNow = cv::getTickCount();
            double sec = (tNow - tStart) / cv::getTickFrequency();
            if (sec >= 1.0) {
                fps = count / sec;
                count = 0;
                tStart = tNow;
                printed = 0;  // allow shape print again next second
            }

            // Display
            cv::putText(frame,
                        "FPS: " + std::to_string((int)fps),
                        cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX,
                        1.0,
                        cv::Scalar(0, 255, 0),
                        2);

            cv::imshow("Camera 640x640", frame);
        }

        if (cv::waitKey(1) == 27) {
            running = false;
            break;
        }
    }

    camThread.join();
    return 0;
}

