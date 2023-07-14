#include "tinyinfer/layer/base_layer.h"
#include "tinyinfer/net/serializer.h"
#include "tinyinfer/net/deserializer.h"

namespace ti {

void BaseLayer::serialize(Serializer& serializer) {
    serialize_internal(serializer);
}
bool BaseLayer::deserialize(Deserializer& deserializer) {
    return deserialize_internal(deserializer);
}

}
