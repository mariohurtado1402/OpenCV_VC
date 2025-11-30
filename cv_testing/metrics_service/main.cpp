#include "httplib.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>

using json = nlohmann::json;

// Almacena la última métrica recibida
struct Metrics {
    double fps = 0.0;
    double latency_ms = 0.0;
    double battery_pct = 0.0;
    int objects = 0;
    std::string source;
    std::string timestamp;
};

std::mutex g_mutex;
std::optional<Metrics> g_latest;

int main() {
    httplib::Server srv;

    // POST /metrics  (Content-Type: application/json)
    srv.Post("/metrics", [](const httplib::Request& req, httplib::Response& res) {
        try {
            auto j = json::parse(req.body);
            Metrics m;
            m.fps = j.value("fps", 0.0);
            m.latency_ms = j.value("latency_ms", 0.0);
            m.battery_pct = j.value("battery_pct", 0.0);
            m.objects = j.value("objects", 0);
            m.source = j.value("source", "");
            m.timestamp = j.value("timestamp", "");

            {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_latest = m;
            }
            res.status = 200;
            res.set_content(R"({"status":"ok"})", "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(std::string(R"({"error":")") + e.what() + R"("})", "application/json");
        }
    });

    // GET /metrics
    srv.Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_latest.has_value()) {
            res.status = 404;
            res.set_content(R"({"error":"no metrics yet"})", "application/json");
            return;
        }
        const Metrics& m = g_latest.value();
        json j = {
            {"fps", m.fps},
            {"latency_ms", m.latency_ms},
            {"battery_pct", m.battery_pct},
            {"objects", m.objects},
            {"source", m.source},
            {"timestamp", m.timestamp},
        };
        res.status = 200;
        res.set_content(j.dump(), "application/json");
    });

    // Raíz simple
    srv.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("Metrics service up. POST /metrics, GET /metrics", "text/plain");
    });

    const std::string host = "0.0.0.0";
    const int port = 8000;
    std::cout << "Listening on http://" << host << ":" << port << "\n";
    srv.listen(host.c_str(), port);
    return 0;
}
