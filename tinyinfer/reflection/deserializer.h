#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/base_layer.h"
#include "tinyinfer/common/layer_factory.h"

namespace ti
{

    class Deserializer
    {
    public:
        bool start(std::string file_path)
        {
            ifs.open(file_path, std::ios::binary);
            CHECK_BOOL_RET(ifs.is_open(), true, "Deserializer file open failed\n")
            unsigned long val;
            ifs.read((char *)&val, sizeof(val));
            CHECK_BOOL_RET(val == NET_START_VAL, true, "Deserializer start failed\n")
            int intval;
            ifs.read((char *)&intval, sizeof(intval));
            layer_num_ = intval;
            cur_layer_ = 0;
            return true;
        }
        bool is_finished()
        {
            return cur_layer_ >= layer_num_;
        }
        bool finish()
        {
            unsigned long val;
            ifs.read((char *)&val, sizeof(val));
            CHECK_BOOL_RET(val == NET_END_VAL, true, "Deserializer finish failed\n")
            ifs.close();
            return true;
        }
        bool begin_layer()
        {
            unsigned long val;
            ifs.read((char *)&val, sizeof(val));
            if (val != LAYER_START_VAL)
            {
                return false;
            }
            cur_layer_++;
            return true;
        }
        bool end_layer()
        {
            unsigned long val;
            ifs.read((char *)&val, sizeof(val));
            if (val != LAYER_END_VAL)
            {
                return false;
            }
            return true;
        }
        std::shared_ptr<BaseLayer> deserialize_one_layer()
        {
            CHECK_RET(begin_layer(), true, nullptr, "deserializer begin failed\n");
            int pos = ifs.tellg();
            LayerType layer_type;
            read(layer_type);
            std::shared_ptr<BaseLayer> layer = LayerFactory::get(layer_type);
            ifs.seekg(pos);
            if (!layer->deserialize(*this))
            {
                std::cerr << "Serialize layer failed\n";
                return nullptr;
            }
            CHECK_RET(end_layer(), true, nullptr, "deserializer end failed\n");
            return layer;
        }
        template <typename T>
        Deserializer &operator()(T &&member)
        {
            if (!this->read(member))
            {
                std::cerr << "err read member\n";
            }
            return *this;
        }
        bool read(LayerType &member)
        {
            int val;
            ifs.read((char *)&val, sizeof(val));
            member = (LayerType)val;
            return true;
        }
        bool read(int &member)
        {
            ifs.read((char *)&member, sizeof(member));
            ;
            return true;
        }
        bool read(unsigned long &member)
        {
            ifs.read((char *)&member, sizeof(member));
            return true;
        }
        bool read(bool &member)
        {
            int val;
            ifs.read((char *)&val, sizeof(val));
            member = val;
            return true;
        }
        bool read(float &member)
        {
            ifs.read((char *)&member, sizeof(member));
            return true;
        }
        bool read(std::string &member)
        {
            int cnt;
            ifs.read((char *)&cnt, sizeof(cnt));
            member.resize(cnt);
            if (cnt != 0)
                ifs.read(member.data(), cnt);
            CHECK_BOOL_RET(member.length(), cnt, "read string length not matched\n");
            return true;
        }
        bool read(std::vector<float> &member)
        {
            int cnt;
            ifs.read((char *)&cnt, sizeof(cnt));
            for (int i = 0; i < cnt; i++)
            {
                float m;
                CHECK_BOOL_RET(read(m), true, "read float failed\n");
                member.emplace_back(std::move(m));
            }
            CHECK_BOOL_RET(member.size(), cnt, "read float array size not matched\n");
            return true;
        }
        bool read(std::vector<unsigned long> &member)
        {
            int cnt;
            ifs.read((char *)&cnt, sizeof(cnt));
            for (int i = 0; i < cnt; i++)
            {
                unsigned long m;
                CHECK_BOOL_RET(read(m), true, "read uint64 failed\n");
                member.emplace_back(std::move(m));
            }
            CHECK_BOOL_RET(member.size(), cnt, "read uint64 array size not matched\n");
            return true;
        }
        bool read(std::vector<std::string> &member)
        {
            int cnt;
            ifs.read((char *)&cnt, sizeof(cnt));
            for (int i = 0; i < cnt; i++)
            {
                std::string m;
                CHECK_BOOL_RET(read(m), true, "read string failed\n");
                member.emplace_back(std::move(m));
            }
            CHECK_BOOL_RET(member.size(), cnt, "read string array size not matched\n");
            return true;
        }
        bool read(std::shared_ptr<Tensor> &tensor)
        {
            tensor.reset(new Tensor());
            int pos = ifs.tellg();
            unsigned long val;
            ifs.read((char *)&val, sizeof(val));
            if (val == EMPTY_VAL)
            {
                return true;
            }
            ifs.seekg(pos);
            return tensor->deserialize(*this);
        }
        template <typename T>
        bool read(T layer_param)
        {
            return layer_param->deserialize_internal(*this);
        }

    private:
        std::ifstream ifs;
        int cur_layer_ = 0;
        int layer_num_ = 0;
    };

}
