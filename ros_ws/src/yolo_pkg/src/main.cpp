#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <filesystem>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

#include <opencv2/opencv.hpp>
#include "net.h"
#include "cpu.h"

// ----------------------------
// CONFIG
// ----------------------------
#define DEVICE "/dev/video0"
#define WIDTH 1280
#define HEIGHT 720
#define CROP_SIZE 640

// Umbrales para filtrar detecciones basura
const float CONF_THRESHOLD = 0.45f;
const float NMS_THRESHOLD = 0.50f;

// Estructura para guardar detecciones
struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct Buffer {
    void* start;
    size_t length;
};

std::atomic<bool> running(true);
std::mutex frameMutex;
cv::Mat latestFrame;

// Buscar los modelos cerca del ejecutable (install/lib/yolo_pkg) o en share/yolo_pkg/models
std::filesystem::path resolve_model_path(const std::string& filename) {
    namespace fs = std::filesystem;
    fs::path exe_path = fs::read_symlink("/proc/self/exe");
    fs::path exe_dir = exe_path.parent_path();

    std::vector<fs::path> candidates = {
        exe_dir / filename,
        exe_dir / "models" / filename,
        exe_dir / ".." / ".." / "share" / "yolo_pkg" / "models" / filename,
        fs::current_path() / filename
    };

    for (const auto& p : candidates) {
        if (fs::exists(p)) return fs::weakly_canonical(p);
    }
    return fs::path(filename);
}

// ======================================================
// CAPTURE THREAD (Igual que antes)
// ======================================================
void captureThread() {
    int fd = open(DEVICE, O_RDWR);
    if (fd < 0) {
        std::cerr << "Error opening V4L2 device\n";
        running = false;
        return;
    }
    v4l2_format fmt {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width  = WIDTH;
    fmt.fmt.pix.height = HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG; 
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    ioctl(fd, VIDIOC_S_FMT, &fmt);

    v4l2_requestbuffers req {};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    ioctl(fd, VIDIOC_REQBUFS, &req);

    Buffer* buffers = new Buffer[req.count];
    for (unsigned i = 0; i < req.count; i++) {
        v4l2_buffer buf {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        ioctl(fd, VIDIOC_QUERYBUF, &buf);
        buffers[i].length = buf.length;
        buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        ioctl(fd, VIDIOC_QBUF, &buf);
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd, VIDIOC_STREAMON, &type);

    while (running) {
        v4l2_buffer buf {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if(ioctl(fd, VIDIOC_DQBUF, &buf) < 0) continue;

        std::vector<uchar> jpegData(buf.bytesused);
        memcpy(jpegData.data(), buffers[buf.index].start, buf.bytesused);
        cv::Mat frame = cv::imdecode(jpegData, cv::IMREAD_COLOR);

        if (!frame.empty()) {
            int startX = std::max(0, (frame.cols - CROP_SIZE) / 2);
            int startY = std::max(0, (frame.rows - CROP_SIZE) / 2);
            cv::Rect roi(startX, startY, CROP_SIZE, CROP_SIZE);
            cv::Mat square = frame(roi).clone();
            
            std::lock_guard<std::mutex> lock(frameMutex);
            latestFrame = square;
        }
        ioctl(fd, VIDIOC_QBUF, &buf);
    }
    ioctl(fd, VIDIOC_STREAMOFF, &type);
    for (unsigned i = 0; i < req.count; i++) munmap(buffers[i].start, buffers[i].length);
    delete[] buffers;
    close(fd);
}

// ======================================================
// MAIN
// ======================================================
ncnn::Net yolo;

int main() {
    const std::filesystem::path param_path = resolve_model_path("best.param");
    const std::filesystem::path bin_path   = resolve_model_path("best.bin");

    // 1. Cargar Modelo
    yolo.opt.num_threads = 4;
    yolo.opt.use_fp16_packed = true;
    yolo.opt.use_fp16_storage = true; 
    
    if (yolo.load_param(param_path.string().c_str()) || yolo.load_model(bin_path.string().c_str())) {
        std::cerr << "Error cargando modelo (revisa rutas a " << param_path << " y " << bin_path << ")\n";
        return -1;
    }
    std::cout << "Modelo cargado desde " << param_path << " / " << bin_path << ". Iniciando captura...\n";

    std::thread camThread(captureThread);

    // Variables FPS
    double fps = 0;
    int count = 0;
    double tStart = cv::getTickCount();

    // Nombres de tus 3 clases (EDÍTALA SEGÚN TU MODELO)
    const std::vector<std::string> class_names = {"Clase0", "Clase1", "Clase2"};

    while (running) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            if (!latestFrame.empty()) frame = latestFrame.clone();
        }

        if (!frame.empty()) {
            // Preprocesamiento NCNN
            ncnn::Mat in = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR, frame.cols, frame.rows);
            const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
            in.substract_mean_normalize(nullptr, norm_vals);

            // Inferencia
            ncnn::Extractor ex = yolo.create_extractor();
            ex.input("in0", in);
            ncnn::Mat out;
            ex.extract("out0", out);

            // ----------------------------------------------------
            // POST-PROCESAMIENTO (Decodificar 8400x7)
            // ----------------------------------------------------
            std::vector<Object> objects;
            
            // out.h = 7 (4 coords + 3 clases), out.w = 8400 anchors
            float* ptr_x = out.row(0); // cx
            float* ptr_y = out.row(1); // cy
            float* ptr_w = out.row(2); // w
            float* ptr_h = out.row(3); // h
            
            // Las clases empiezan en la fila 4
            // Como tienes 3 clases, revisamos filas 4, 5 y 6
            
            for (int i = 0; i < out.w; i++) {
                // Encontrar la clase con mayor probabilidad
                float max_score = -1000.f;
                int label = -1;

                // Iteramos sobre las 3 clases
                for (int k = 0; k < 3; k++) {
                   float score = out.row(4 + k)[i];
                   if (score > max_score) {
                       max_score = score;
                       label = k;
                   }
                }

                if (max_score >= CONF_THRESHOLD) {
                    float cx = ptr_x[i] * frame.cols; // des-normalizar si es necesario, pero YOLOv8 suele dar pixel coords
                    float cy = ptr_y[i] * frame.rows; 
                    float w  = ptr_w[i] * frame.cols; 
                    float h  = ptr_h[i] * frame.rows;

                    // NOTA: Si tu modelo fue exportado sin normalizar, quita "* frame.cols". 
                    // Si las cajas salen gigantes, es porque el modelo ya entrega pixeles y no 0-1.
                    // Probaremos asumiendo pixeles directos primero:
                    
                    /* Si las cajas son enormes, usa esto: */
                    cx = ptr_x[i]; cy = ptr_y[i]; w = ptr_w[i]; h = ptr_h[i];

                    Object obj;
                    obj.rect.x = cx - w / 2;
                    obj.rect.y = cy - h / 2;
                    obj.rect.width = w;
                    obj.rect.height = h;
                    obj.label = label;
                    obj.prob = max_score;
                    objects.push_back(obj);
                }
            }

            // Non-Maximum Suppression (NMS) usando OpenCV
            std::vector<int> indices;
            std::vector<cv::Rect> bboxes;
            std::vector<float> scores;
            for (auto& obj : objects) {
                bboxes.push_back(obj.rect);
                scores.push_back(obj.prob);
            }
            cv::dnn::NMSBoxes(bboxes, scores, CONF_THRESHOLD, NMS_THRESHOLD, indices);

            // ----------------------------------------------------
            // DIBUJAR
            // ----------------------------------------------------
            for (int idx : indices) {
                const Object& obj = objects[idx];
                cv::rectangle(frame, obj.rect, cv::Scalar(0, 255, 0), 2);
                
                std::string text = class_names[obj.label] + " " + cv::format("%.2f", obj.prob);
                
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                cv::rectangle(frame, cv::Rect(cv::Point(obj.rect.x, obj.rect.y - labelSize.height),
                                              cv::Size(labelSize.width, labelSize.height + baseLine)),
                              cv::Scalar(0, 255, 0), -1);
                              
                cv::putText(frame, text, cv::Point(obj.rect.x, obj.rect.y),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            }

            // FPS
            count++;
            double tNow = cv::getTickCount();
            if ((tNow - tStart) / cv::getTickFrequency() >= 1.0) {
                fps = count / ((tNow - tStart) / cv::getTickFrequency());
                count = 0; tStart = tNow;
            }
            cv::putText(frame, "FPS: " + std::to_string((int)fps), cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            cv::imshow("Deteccion YOLO", frame);
        }

        if (cv::waitKey(1) == 27) { // ESC para salir
            running = false;
        }
    }

    camThread.join();
    return 0;
}
