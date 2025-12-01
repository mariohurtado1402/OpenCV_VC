#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <filesystem>
#include <set>
#include <cstdlib>
#include <fstream>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

#include <opencv2/opencv.hpp>
#include "net.h"
#include "cpu.h"
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>

// ----------------------------
// CONFIG
// ----------------------------
#define DEVICE "/dev/video0"
#define WIDTH 640
#define HEIGHT 480
#define CROP_SIZE 480  // recortamos a 480x480 (modelo entrenado a 480)

// Umbrales para filtrar detecciones basura
const float CONF_THRESHOLD = 0.45f;
const float NMS_THRESHOLD = 0.50f;
const int CENTER_TOL_PX = 60;
const float SMOOTH_ALPHA = 0.25f;
const double PUBLISH_HZ = 20.0;
const double KEEPALIVE_SEC = 0.30;
const double SEARCH_ROTATION_SEC = 4.0;
const char SEARCH_DIRECTION = 'L'; // 'L' o 'R'
const double LADDER_FORWARD_SEC = 3.0;
const double LADDER_TURN_SEC = 2.0;
const int LADDER_REPEATS = 3;
const double LADDER_EXPAND_STEP_SEC = 0.0;
const int MAX_CAPTURES = 3;  // detener tras capturar 3 clases distintas
const char* DRIVE_UPLOADER = "/home/mario/OpenCV_VC/cv_testing/upload_to_drive.py";
const char* CAPTURE_DIR = "/home/mario/OpenCV_VC/ros_ws";
// Gate opcional por topic /start_gate (Bool); si es false se mantiene en S

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
bool start_ready = true;
bool use_start_topic = false;
float battery_level = -1.0f;

// Buscar los modelos cerca del ejecutable (install/lib/yolo_pkg) o en share/yolo_pkg/models
std::filesystem::path resolve_model_path(const std::string& filename) {
    namespace fs = std::filesystem;
    fs::path exe_path = fs::read_symlink("/proc/self/exe");
    fs::path exe_dir = exe_path.parent_path();
    fs::path src_dir = fs::path(__FILE__).parent_path().parent_path(); // .../yolo_pkg/src -> package root
    const char* env_dir = std::getenv("YOLO_MODEL_DIR");

    std::vector<fs::path> candidates = {
        exe_dir / filename,
        exe_dir / "models" / filename,
        exe_dir / ".." / ".." / "share" / "yolo_pkg" / "models" / filename,
        exe_dir / ".." / ".." / ".." / "share" / "yolo_pkg" / "models" / filename, // symlink install case
        src_dir / "models" / filename,
        fs::current_path() / filename
    };
    if (env_dir && *env_dir) {
        candidates.insert(candidates.begin(), fs::path(env_dir) / filename);
    }

    for (const auto& p : candidates) {
        if (fs::exists(p)) return fs::weakly_canonical(p);
    }
    std::cerr << "No se encontró el modelo " << filename << ". Probados:\n";
    for (const auto& p : candidates) std::cerr << "  " << p << "\n";
    return fs::path(filename);
}


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
            // Asegura un recorte cuadrado válido incluso si la cámara no es 1:1
            const int roi_size = std::min({frame.cols, frame.rows, CROP_SIZE});
            int startX = std::max(0, (frame.cols - roi_size) / 2);
            int startY = std::max(0, (frame.rows - roi_size) / 2);
            cv::Rect roi(startX, startY, roi_size, roi_size);
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
    rclcpp::init(0, nullptr);
    auto node = std::make_shared<rclcpp::Node>("yolo_tracker");
    auto pub_cmd = node->create_publisher<std_msgs::msg::String>("/cmd/vision", 10);
    auto pub_stats = node->create_publisher<std_msgs::msg::Float32MultiArray>("/vision/stats", 10);
    auto sub_start = node->create_subscription<std_msgs::msg::Bool>("/start_gate", 10,
        [](std_msgs::msg::Bool::SharedPtr msg){
            start_ready = msg->data;
            use_start_topic = true;
        });
    auto sub_batt = node->create_subscription<std_msgs::msg::Float32>("/battery", 10,
        [](std_msgs::msg::Float32::SharedPtr msg){
            battery_level = msg->data;
        });

    // Prefer new 480x480 export; fallback a best.param/bin
    std::filesystem::path param_path = resolve_model_path("best.param");
    std::filesystem::path bin_path   = resolve_model_path("best.bin");
    if (!std::filesystem::exists(param_path) || !std::filesystem::exists(bin_path)) {
        param_path = resolve_model_path("best.param");
        bin_path   = resolve_model_path("best.bin");
    }

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

    std::vector<std::string> class_names = {"Dinosaurio"};

    // Seguimiento/búsqueda
    bool search_completed = false;
    auto search_start = std::chrono::steady_clock::now();
    bool ladder_active = false;
    std::deque<std::pair<char, double>> ladder_plan;
    char ladder_cmd = 'S';
    auto ladder_cmd_until = std::chrono::steady_clock::now();
    cv::Point smoothed_center(-1, -1);
    char last_cmd_sent = '\0';
    auto last_pub = std::chrono::steady_clock::now();
    rclcpp::Rate loop_rate(PUBLISH_HZ);
    std::set<std::string> captured_labels;
    bool stop_requested = false;
    float max_seen_count = 0.0f;

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

            // Validar la forma del tensor de salida antes de indexar
            if (out.empty() || out.h < 5 || out.w == 0) {
                std::cerr << "Salida inesperada del modelo. dims=" << out.dims
                          << " w=" << out.w << " h=" << out.h << " c=" << out.c << "\n";
                cv::imshow("Deteccion YOLO", frame);
                if (cv::waitKey(1) == 27) running = false;
                continue;
            }

            // ----------------------------------------------------
            // POST-PROCESAMIENTO (decodifica salida dinámica)
            // ----------------------------------------------------
            std::vector<Object> objects;
            const int num_classes = out.h - 4; // filas = 4 coords + clases
            if (num_classes <= 0) {
                std::cerr << "Salida inesperada: num_classes <= 0 (h=" << out.h << ")\n";
            } else {
                // Rellena nombres si el modelo tiene más clases de las declaradas
                if ((int)class_names.size() < num_classes) {
                    for (int i = class_names.size(); i < num_classes; ++i) {
                        class_names.push_back("Clase" + std::to_string(i));
                    }
                }

                float* ptr_x = out.row(0); // cx
                float* ptr_y = out.row(1); // cy
                float* ptr_w = out.row(2); // w
                float* ptr_h = out.row(3); // h

                for (int i = 0; i < out.w; i++) {
                    float max_score = -1000.f;
                    int label = 0;

                    if (num_classes == 1) {
                        max_score = out.row(4)[i];
                        label = 0;
                    } else {
                        for (int k = 0; k < num_classes; k++) {
                            float score = out.row(4 + k)[i];
                            if (score > max_score) {
                                max_score = score;
                                label = k;
                            }
                        }
                    }

                    if (max_score >= CONF_THRESHOLD) {
                        // El modelo ya entrega coordenadas en pixeles (por pnnx anchors); no re-escalar.
                        float cx = ptr_x[i];
                        float cy = ptr_y[i];
                        float w  = ptr_w[i];
                        float h  = ptr_h[i];

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
            // LOGICA DE SEGUIMIENTO / BUSQUEDA
            // ----------------------------------------------------
            auto now = std::chrono::steady_clock::now();
            bool searching = false;
            if (!search_completed) {
                double elapsed = std::chrono::duration<double>(now - search_start).count();
                searching = elapsed < SEARCH_ROTATION_SEC;
                if (!searching) {
                    search_completed = true;
                    ladder_active = false;
                    ladder_plan.clear();
                    std::cout << "[Vision] Búsqueda 360 completada, iniciando ladder si no hay objetivo.\n";
                }
            }

            // Ladder plan management
            auto ensure_ladder = [&](double forward_sec, double turn_sec, int repeats, double expand_step) {
                if (ladder_active) return;
                ladder_plan.clear();
                double fwd = forward_sec;
                for (int i = 0; i < repeats; ++i) {
                    ladder_plan.push_back({'F', fwd});
                    ladder_plan.push_back({'L', turn_sec});
                    ladder_plan.push_back({'R', 2 * turn_sec});
                    ladder_plan.push_back({'L', turn_sec});
                    fwd += std::max(0.0, expand_step);
                }
                ladder_active = true;
                ladder_cmd = 'S';
                ladder_cmd_until = now;
                std::cout << "[Vision] Ladder iniciada\n";
            };

            auto advance_ladder = [&](auto tnow) {
                if (!ladder_active) return false;
                if (tnow >= ladder_cmd_until) {
                    if (ladder_plan.empty()) {
                        ladder_active = false;
                        ladder_cmd = 'S';
                        return false;
                    }
                    auto [c, dur] = ladder_plan.front();
                    ladder_plan.pop_front();
                    ladder_cmd = c;
                    auto dur_cast = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(dur));
                    ladder_cmd_until = tnow + dur_cast;
                    std::cout << "[Vision] Ladder cmd " << c << " " << dur << "s\n";
                }
                return true;
            };

            char out_cmd = 'S';
            bool has_det = !indices.empty();

            // Gate por /start_gate: si no está listo, mandar S y no procesar
            if (use_start_topic && !start_ready) {
                ladder_active = false;
                ladder_plan.clear();
                has_det = false;
                out_cmd = 'S';
            }

            if (stop_requested) {
                out_cmd = 'S';
                ladder_active = false;
                ladder_plan.clear();
            } else
            if (has_det) {
                const Object& obj = objects[indices[0]];
                int cx = static_cast<int>(obj.rect.x + obj.rect.width * 0.5f);
                int cy = static_cast<int>(obj.rect.y + obj.rect.height * 0.5f);
                if (smoothed_center.x < 0) smoothed_center = cv::Point(cx, cy);
                smoothed_center.x = static_cast<int>((1 - SMOOTH_ALPHA) * smoothed_center.x + SMOOTH_ALPHA * cx);
                smoothed_center.y = static_cast<int>((1 - SMOOTH_ALPHA) * smoothed_center.y + SMOOTH_ALPHA * cy);

                int dx = smoothed_center.x - (frame.cols / 2);
                std::string cls = (obj.label >= 0 && obj.label < (int)class_names.size())
                                    ? class_names[obj.label]
                                    : ("Clase" + std::to_string(obj.label));
                // Ignorar clases ya capturadas
                if (captured_labels.count(cls) > 0) {
                    has_det = false;
                } else {
                    if (searching) {
                        out_cmd = SEARCH_DIRECTION;
                    } else {
                        if (std::abs(dx) <= CENTER_TOL_PX) {
                            // Centrado: capturar, subir, marcar y reiniciar búsqueda
                            std::string ts = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());
                            std::string filename = std::string("capture_") + cls + "_" + ts + ".jpg";
                            std::string fullpath = std::string(CAPTURE_DIR) + "/" + filename;
                            try {
                                cv::imwrite(fullpath, frame);
                                std::cout << "[Capture] Guardada " << fullpath << "\n";
                                std::string cmd = std::string("python3 ") + DRIVE_UPLOADER + " \"" + fullpath + "\"";
                                int rc = std::system(cmd.c_str());
                                std::cout << "[Capture] Upload rc=" << rc << "\n";
                            } catch (const std::exception& e) {
                                std::cerr << "[Capture] Error guardando/subiendo: " << e.what() << "\n";
                            }
                            captured_labels.insert(cls);
                            if ((int)captured_labels.size() >= MAX_CAPTURES) {
                                stop_requested = true;
                                std::cout << "[Capture] Max capturas alcanzado, deteniendo.\n";
                            } else {
                                // Reinicia búsqueda amplia
                                search_completed = false;
                                search_start = now;
                                ladder_active = false;
                                ladder_plan.clear();
                            }
                            // Reinicia búsqueda amplia
                            out_cmd = 'S';
                        } else if (dx > CENTER_TOL_PX) out_cmd = 'R';
                        else if (dx < -CENTER_TOL_PX) out_cmd = 'L';
                        else out_cmd = 'S';
                    }

                    ladder_active = false;
                    ladder_plan.clear();

                    std::cout << "[YOLO] Detectado " << cls
                              << " prob=" << obj.prob
                              << " cmd=" << out_cmd
                              << " center=(" << cx << "," << cy << ")\n";
                }
            } else {
                // Sin detección
                if (!searching && search_completed && !stop_requested) {
                    ensure_ladder(LADDER_FORWARD_SEC, LADDER_TURN_SEC, LADDER_REPEATS, LADDER_EXPAND_STEP_SEC);
                    if (advance_ladder(now)) {
                        out_cmd = ladder_cmd;
                    }
                }
                if (searching) {
                    out_cmd = SEARCH_DIRECTION;
                }
            }

            // Publicar comando con keepalive
            auto publish_cmd = [&](char c) {
                auto msg = std_msgs::msg::String();
                msg.data = std::string(1, c);
                pub_cmd->publish(msg);
            };

            double since_last = std::chrono::duration<double>(now - last_pub).count();
            if (out_cmd != last_cmd_sent || since_last >= KEEPALIVE_SEC) {
                publish_cmd(out_cmd);
                last_cmd_sent = out_cmd;
                last_pub = now;
                std::cout << "[CMD] /cmd/vision -> " << out_cmd << "\n";
            }

            // Publicar métricas
            std_msgs::msg::Float32MultiArray stats;
            float latency = (fps > 0) ? (1.0f / static_cast<float>(fps)) : 0.0f;
            float battery = battery_level;
            float current_seen = static_cast<float>(std::min<int>(indices.size(), 3));
            float captures_seen = static_cast<float>(std::min<int>((int)captured_labels.size(), 3));
            if (current_seen > max_seen_count) max_seen_count = current_seen;
            if (captures_seen > max_seen_count) max_seen_count = captures_seen;
            float obj_count = max_seen_count;
            stats.data = {static_cast<float>(fps), latency, battery, obj_count};
            pub_stats->publish(stats);

            // FPS
            count++;
            double tNow = cv::getTickCount();
            if ((tNow - tStart) / cv::getTickFrequency() >= 1.0) {
                fps = count / ((tNow - tStart) / cv::getTickFrequency());
                count = 0; tStart = tNow;
                std::cout << "[YOLO] FPS=" << fps << "\n";
            }

            // Visualización básica
            for (int idx : indices) {
                const Object& obj = objects[idx];
                cv::rectangle(frame, obj.rect, cv::Scalar(0, 255, 0), 2);
                std::string cls = (obj.label >= 0 && obj.label < (int)class_names.size())
                                    ? class_names[obj.label]
                                    : ("Clase" + std::to_string(obj.label));
                std::string text = cls + " " + cv::format("%.2f", obj.prob);
                cv::putText(frame, text, cv::Point(obj.rect.x, obj.rect.y - 4),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
            cv::imshow("Deteccion YOLO", frame);
            if (cv::waitKey(1) == 27) {
                running = false;
            }
        }

        rclcpp::spin_some(node);
        loop_rate.sleep();
    }

    camThread.join();
    rclcpp::shutdown();
    return 0;
}
