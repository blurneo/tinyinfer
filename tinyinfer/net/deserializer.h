#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include "tinyinfer/common/check_macro.h"

namespace ti {

class Deserializer {
 public:
    bool start(std::string file_path) {
        ifs.open(file_path);
        CHECK_BOOL_RET(ifs.is_open(), true, "Deserializer file open failed\n")
        sstream << ifs.rdbuf();
        return true;
    }
    void finish() {
        ifs.close();
    }
    bool begin() {
        std::string flag;
        sstream >> flag;
        if (flag != "Layer:") {
            return false;
        }
        return true;
    }
    bool end() {
        // ifs << "\n";
    }
    template<typename T>
    std::optional<Deserializer&> operator()(std::string field_name, T& member) {
        std::string flag;
        sstream >> flag;
        if (flag != field_name) {
            std::cerr << "err read fieldname\n";
            // return std::nullopt;
        }
        if (!this->read(member)) {
            std::cerr << "err read member\n";
            // return std::nullopt;
        }
        return *this;
    }
    bool read(int &member) {
        std::string flag;
        sstream >> flag;
        CHECK_BOOL_RET(flag == "int:", true, "read int failed\n")
        sstream >> member;
        return true;
    }
    bool read(float &member) {
        std::string flag;
        sstream >> flag;
        CHECK_BOOL_RET(flag == "f4:", true, "read float failed\n")
        sstream >> member;
        return true;
    }
    bool read(std::string &member) {
        std::string flag;
        sstream >> flag;
        CHECK_BOOL_RET(flag == "str:", true, "read string failed\n")
        sstream >> member;
        return true;
    }
    bool read(const std::vector<float> &member) {
        std::string flag;
        sstream >> flag;
        CHECK_BOOL_RET(flag == "f4[]:", true, "read float array failed\n")
        for (auto m : member) {
            CHECK_BOOL_RET(read(m), true, "read float failed\n");
        }
        return true;
    }
    bool read(const std::vector<std::string> &member) {
        std::string flag;
        sstream >> flag;
        CHECK_BOOL_RET(flag == "str[]:", true, "read str array failed\n")
        for (auto m : member) {
            CHECK_BOOL_RET(read(m), true, "read string failed\n");
        }
        return true;
    }
 private:
    std::ifstream ifs;
    std::stringstream sstream;
};

}
