#include "tinyinfer/common/tensor.h"
#include "tinyinfer/net/serializer.h"
#include "tinyinfer/net/deserializer.h"

namespace ti {

void Tensor::serialize(Serializer& serializer) {
    serialize_internal(serializer);
}

bool Tensor::deserialize(Deserializer& deserializer) {
    return deserialize_internal(deserializer);
}
}
