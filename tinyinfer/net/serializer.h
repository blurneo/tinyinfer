#pragma once

#include <vector>
#include <string>
#include <fstream>
#include "tinyinfer/common/check_macro.h"

namespace ti {

class Serializer {
 public:
    bool start(std::string file_path) {
        ofs.open(file_path);
        CHECK_BOOL_RET(ofs.is_open(), true, "Serializer file open failed\n")
        return true;
    }
    void finish() {
        ofs.close();
    }
    void begin() {
        ofs << "Layer: ";
    }
    void end() {
        ofs << "\n";
    }
    template<typename T>
    Serializer& operator()(std::string field_name, T member) {
        ofs << field_name << " ";
        this->write(member);
        return *this;
    }
    void write(int member) {
        ofs << "int: ";
        ofs << member << " ";
    }
    void write(int member) {
        ofs << "f4: ";
        ofs << member << " ";
    }
    void write(std::string member) {
        ofs << "str, " << member.length() << ": ";
        ofs << member << " ";
    }
    void write(const std::vector<float> &member) {
        ofs << "f4[], " << member.size() << ": ";
        for (auto m : member) write(m);
    }
    void write(const std::vector<std::string> &member) {
        ofs << "str[], " << member.size() << ": ";
        for (auto m : member) ofs << m << " ";
    }
 private:
    std::ofstream ofs;
};

}
