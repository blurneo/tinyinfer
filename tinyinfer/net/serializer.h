#pragma once

#include <vector>
#include <string>
#include <fstream>

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
        ofs << member << " ";
    }
    void write(std::string member) {
        ofs << member << " ";
    }
    void write(const std::vector<float> &member) {
        for (auto m : member) ofs << m << " ";
    }
    void write(const std::vector<std::string> &member) {
        for (auto m : member) ofs << m << " ";
    }
 private:
    std::ofstream ofs;
};

}