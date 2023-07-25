#include "tinyinfer/common/base_layer.h"
#include "tinyinfer/reflection/serializer.h"
#include "tinyinfer/reflection/deserializer.h"

namespace ti
{

    void BaseLayer::serialize(Serializer &serializer)
    {
        serialize_internal(serializer);
    }
    bool BaseLayer::deserialize(Deserializer &deserializer)
    {
        return deserialize_internal(deserializer);
    }

}
