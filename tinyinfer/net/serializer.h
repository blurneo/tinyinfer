#pragma once

#include <vector>
#include <string>
#include <fstream>
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/layer/base_layer.h"

namespace ti {

class Serializer {
 public:
    bool start(std::string file_path, int layer_num) {
        ofs.open(file_path);
        CHECK_BOOL_RET(ofs.is_open(), true, "Serializer file open failed\n")
        ofs << "NetStart: " << layer_num << "\n";
        return true;
    }
    void finish() {
        ofs << "NetEnd\n";
        ofs.close();
    }
    void begin_layer() {
        ofs << "Layer: ";
    }
    void end_layer() {
        ofs << "\n";
    }
    void serialize_one_layer(std::shared_ptr<BaseLayer> layer) {
        begin_layer();
        layer->serialize(*this);
        end_layer();
    }
    template<typename T>
    Serializer& operator()(std::string field_name, T member) {
        ofs << field_name << " ";
        this->write(member);
        return *this;
    }
    void write(LayerType member) {
        ofs << "int: ";
        ofs << member << " ";
    }
    void write(bool member) {
        ofs << "bool: ";
        ofs << member << " ";
    }
    void write(int member) {
        ofs << "int: ";
        ofs << member << " ";
    }
    void write(float member) {
        ofs << "f4: ";
        ofs << member << " ";
    }
    void write(unsigned long member) {
        ofs << "uint64: ";
        ofs << member << " ";
    }
    void write(std::string member) {
        ofs << "str: " << member.length() << " ";
        ofs << member << " ";
    }
    void write(const std::vector<float> &member) {
        ofs << "f4[]: " << member.size() << " ";
        for (auto m : member) write(m);
    }
    void write(const std::vector<unsigned long> &member) {
        ofs << "uint64[]: " << member.size() << " ";
        for (auto m : member) write(m);
    }
    void write(const std::vector<std::string> &member) {
        ofs << "str[]: " << member.size() << " ";
        for (auto m : member) write(m);
    }
    void write(std::shared_ptr<Tensor> tensor) {
        if (tensor->get_values().empty()) {
            ofs << "empty";
            return;
        }
        tensor->serialize(*this);
    }
    template<typename T>
    void write(T layer_param) {
        layer_param->serialize_internal(*this);
    }
 private:
    std::ofstream ofs;
};

}
