#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/base_layer.h"
#include "tinyinfer/layer/layer_factory.h"

namespace ti {

class Deserializer {
 public:
    bool start(std::string file_path) {
        ifs.open(file_path);
        CHECK_BOOL_RET(ifs.is_open(), true, "Deserializer file open failed\n")
        sstream << ifs.rdbuf();
        sstream >> flag;
        CHECK_BOOL_RET(flag == "NetStart:", true, "Deserializer start failed\n")
        sstream >> layer_num_;
        cur_layer_ = 0;
        return true;
    }
    bool is_finished() {
        return cur_layer_ >= layer_num_;
    }
    bool finish() {
        sstream >> flag;
        CHECK_BOOL_RET(flag == "NetEnd", true, "Deserializer finish failed\n")
        ifs.close();
        return true;
    }
    bool begin_layer() {
        sstream >> flag;
        if (flag != "Layer:") {
            return false;
        }
        cur_layer_++;
        return true;
    }
    bool end_layer() {
        // ifs << "\n";
        return true;
    }
    std::shared_ptr<BaseLayer> deserialize_one_layer() {
        CHECK_RET(begin_layer(), true, nullptr, "deserializer begin failed\n");
        int pos = sstream.tellg();
        LayerType layer_type;
        this->operator()("layer_type", layer_type);
        std::shared_ptr<BaseLayer> layer = LayerFactory::get(layer_type);
        sstream.seekg(pos);
        if (!layer->deserialize(*this)) {
            std::cerr << "Serialize layer failed\n";
            return nullptr;
        }
        CHECK_RET(end_layer(), true, nullptr, "deserializer end failed\n");
        return layer;
    }
    template<typename T>
    Deserializer& operator()(std::string field_name, T&& member) {
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
    bool read(LayerType &member) {
        sstream >> flag;
        CHECK_BOOL_RET(flag == "int:", true, "read int failed\n")
        int val;
        sstream >> val;
        member = (LayerType)val;
        return true;
    }
    bool read(int &member) {
        sstream >> flag;
        CHECK_BOOL_RET(flag == "int:", true, "read int failed\n")
        sstream >> member;
        return true;
    }
    bool read(unsigned long &member) {
        sstream >> flag;
        CHECK_BOOL_RET(flag == "uint64:", true, "read uint64 failed\n")
        sstream >> member;
        return true;
    }
    bool read(bool &member) {
        sstream >> flag;
        CHECK_BOOL_RET(flag == "bool:", true, "read bool failed\n")
        sstream >> member;
        return true;
    }
    bool read(float &member) {
        sstream >> flag;
        CHECK_BOOL_RET(flag == "f4:", true, "read float failed\n")
        sstream >> member;
        return true;
    }
    bool read(std::string &member) {
        sstream >> flag;
        CHECK_BOOL_RET(flag == "str:", true, "read string failed\n")
        int cnt;
        sstream >> cnt;
        if (cnt != 0) sstream >> member;
        CHECK_BOOL_RET(member.length(), cnt, "read string length not matched\n");
        return true;
    }
    bool read(std::vector<float> &member) {
        sstream >> flag;
        CHECK_BOOL_RET(flag == "f4[]:", true, "read float array failed\n")
        int cnt;
        sstream >> cnt;
        for (int i = 0; i < cnt; i++) {
            float m;
            CHECK_BOOL_RET(read(m), true, "read float failed\n");
            member.emplace_back(std::move(m));
        }
        CHECK_BOOL_RET(member.size(), cnt, "read float array size not matched\n");
        return true;
    }
    bool read(std::vector<unsigned long> &member) {
        sstream >> flag;
        CHECK_BOOL_RET(flag == "uint64[]:", true, "read uin64 array failed\n")
        int cnt;
        sstream >> cnt;
        for (int i = 0; i < cnt; i++) {
            unsigned long m;
            CHECK_BOOL_RET(read(m), true, "read uint64 failed\n");
            member.emplace_back(std::move(m));
        }
        CHECK_BOOL_RET(member.size(), cnt, "read uint64 array size not matched\n");
        return true;
    }
    bool read(std::vector<std::string> &member) {
        sstream >> flag;
        CHECK_BOOL_RET(flag == "str[]:", true, "read str array failed\n")
        int cnt;
        sstream >> cnt;
        for (int i = 0; i < cnt; i++) {
            std::string m;
            CHECK_BOOL_RET(read(m), true, "read string failed\n");
            member.emplace_back(std::move(m));
        }
        CHECK_BOOL_RET(member.size(), cnt, "read string array size not matched\n");
        return true;
    }
    bool read(std::shared_ptr<Tensor> &tensor) {
        tensor.reset(new Tensor());
        int pos = sstream.tellg();
        sstream >> flag;
        if (flag == "empty") {
            return true;
        }
        sstream.seekg(pos);
        return tensor->deserialize(*this);
    }
    template<typename T>
    bool read(T layer_param) {
        return layer_param->deserialize_internal(*this);
    }
 private:
    std::ifstream ifs;
    std::stringstream sstream;
    std::string flag;
    int cur_layer_ = 0;
    int layer_num_ = 0;
};

}