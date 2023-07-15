#pragma once

#include <vector>
#include <string>
#include <fstream>
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/base_layer.h"
#include "tinyinfer/reflection/serialize_macro.h"

namespace ti {

class Serializer {
 public:
    bool start(std::string file_path, int layer_num) {
        ofs.open(file_path, std::ios::binary);
        CHECK_BOOL_RET(ofs.is_open(), true, "Serializer file open failed\n")
        ofs.write((char*)(&NET_START_VAL), sizeof(NET_START_VAL));
        ofs.write((char*)(&layer_num), sizeof(layer_num));
        return true;
    }
    void finish() {
        ofs.write((char*)(&NET_END_VAL), sizeof(NET_END_VAL));
        ofs.close();
    }
    void begin_layer() {
        ofs.write((char*)(&LAYER_START_VAL), sizeof(LAYER_START_VAL));
    }
    void end_layer() {
        ofs.write((char*)(&LAYER_END_VAL), sizeof(LAYER_END_VAL));
    }
    void serialize_one_layer(std::shared_ptr<BaseLayer> layer) {
        begin_layer();
        layer->serialize(*this);
        end_layer();
    }
    template<typename T>
    Serializer& operator()(std::string field_name, T member) {
        this->write(member);
        return *this;
    }
    void write(LayerType member) {
        int val = member;
        ofs.write((char*)&val, sizeof(val));
    }
    void write(bool member) {
        int val = member;
        ofs.write((char*)&val, sizeof(val));
    }
    void write(int member) {
        ofs.write((char*)&member, sizeof(member));
    }
    void write(float member) {
        ofs.write((char*)&member, sizeof(member));
    }
    void write(unsigned long member) {
        ofs.write((char*)&member, sizeof(member));
    }
    void write(std::string member) {
        int length = member.length();
        ofs.write((char*)&length, sizeof(length));
        ofs.write(member.c_str(), length);
    }
    void write(const std::vector<float> &member) {
        int length = member.size();
        ofs.write((char*)&length, sizeof(length));
        for (auto m : member) write(m);
    }
    void write(const std::vector<unsigned long> &member) {
        int length = member.size();
        ofs.write((char*)&length, sizeof(length));
        for (auto m : member) write(m);
    }
    void write(const std::vector<std::string> &member) {
        int length = member.size();
        ofs.write((char*)&length, sizeof(length));
        for (auto m : member) write(m);
    }
    void write(std::shared_ptr<Tensor> tensor) {
        if (tensor->get_values().empty()) {
            ofs.write((char*)&EMPTY_VAL, sizeof(EMPTY_VAL));
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
