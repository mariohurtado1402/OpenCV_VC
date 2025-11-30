#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Ajusta estas rutas e IDs a tu entorno
const std::string kTokenFile = "/home/mario/OpenCV_VC/cv_testing/token.json";
const std::string kClientFile = "/home/mario/OpenCV_VC/cv_testing/credentials.json";
const std::string kDriveFolderId = "1EdP-E2N8aJFVE3lpVX8mbdzueAb6ceeB";

struct TokenInfo {
    std::string client_id;
    std::string client_secret;
    std::string refresh_token;   // puede venir vacío
    std::string access_token;    // fallback si no hay refresh_token
};

// Lee archivo completo en un string
std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("No se pudo abrir: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Extrae client_id/client_secret desde credentials.json (formato OAuth web)
TokenInfo load_tokens() {
    using json = nlohmann::json;
    json creds = json::parse(read_file(kClientFile));
    json token = json::parse(read_file(kTokenFile));

    TokenInfo info;
    info.client_id = creds["installed"]["client_id"].get<std::string>();
    info.client_secret = creds["installed"]["client_secret"].get<std::string>();
    // refresh_token puede ser null/empty; access_token puede servir mientras no expire
    if (token.contains("refresh_token") && !token["refresh_token"].is_null()) {
        info.refresh_token = token["refresh_token"].get<std::string>();
    }
    if (token.contains("access_token") && !token["access_token"].is_null()) {
        info.access_token = token["access_token"].get<std::string>();
    }
    if (info.client_id.empty() || info.client_secret.empty()) {
        throw std::runtime_error("Faltan client_id/client_secret");
    }
    if (info.refresh_token.empty() && info.access_token.empty()) {
        throw std::runtime_error("No hay refresh_token ni access_token en token.json. Reautentica con pydrive.");
    }
    return info;
}

// Callback para capturar respuesta HTTP
size_t write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* buf = static_cast<std::string*>(userdata);
    buf->append(ptr, size * nmemb);
    return size * nmemb;
}

// Pide nuevo access_token usando refresh_token
std::string fetch_access_token(const TokenInfo& info) {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl_easy_init failed");

    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, "https://oauth2.googleapis.com/token");
    curl_easy_setopt(curl, CURLOPT_POST, 1L);

    std::ostringstream body;
    body << "client_id=" << curl_easy_escape(curl, info.client_id.c_str(), 0)
         << "&client_secret=" << curl_easy_escape(curl, info.client_secret.c_str(), 0)
         << "&refresh_token=" << curl_easy_escape(curl, info.refresh_token.c_str(), 0)
         << "&grant_type=refresh_token";
    auto body_str = body.str();
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_str.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    if (res != CURLE_OK) {
        throw std::runtime_error(std::string("curl error: ") + curl_easy_strerror(res));
    }

    auto json_resp = nlohmann::json::parse(response);
    if (!json_resp.contains("access_token")) {
        throw std::runtime_error("No access_token en respuesta: " + response);
    }
    return json_resp["access_token"].get<std::string>();
}

// Sube un CSV mínimo a Drive (multipart upload)
void upload_csv(const std::string& access_token) {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl_easy_init failed");

    const std::string metadata = R"({"name":"test_cpp_drive.csv","parents":[")" + kDriveFolderId + R"("]})";
    const std::string csv_data = "columna\nvalor_cpp\n";

    std::string boundary = "----driveboundary123456";
    std::ostringstream payload;
    payload << "--" << boundary << "\r\n";
    payload << "Content-Type: application/json; charset=UTF-8\r\n\r\n";
    payload << metadata << "\r\n";
    payload << "--" << boundary << "\r\n";
    payload << "Content-Type: text/csv\r\n\r\n";
    payload << csv_data << "\r\n";
    payload << "--" << boundary << "--\r\n";
    std::string payload_str = payload.str();

    std::string url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart";
    struct curl_slist* headers = nullptr;
    std::string auth = "Authorization: Bearer " + access_token;
    headers = curl_slist_append(headers, auth.c_str());
    std::string content_type = "Content-Type: multipart/related; boundary=" + boundary;
    headers = curl_slist_append(headers, content_type.c_str());

    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_str.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, payload_str.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        throw std::runtime_error(std::string("curl error: ") + curl_easy_strerror(res));
    }
    if (http_code >= 300) {
        throw std::runtime_error("HTTP error " + std::to_string(http_code) + ": " + response);
    }
    std::cout << "✅ Subido test_cpp_drive.csv\n";
}

int main() {
    try {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        TokenInfo info = load_tokens();

        std::string access_token;
        if (!info.refresh_token.empty()) {
            access_token = fetch_access_token(info);
        } else {
            // Usamos el access_token existente (hasta que expire)
            access_token = info.access_token;
            if (access_token.empty()) {
                throw std::runtime_error("No hay access_token disponible.");
            }
            std::cout << "Usando access_token existente (sin refresh_token)\n";
        }

        upload_csv(access_token);
        curl_global_cleanup();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        curl_global_cleanup();
        return 1;
    }
    return 0;
}
