#pragma once

#define DEFINE_SERIALIZE_MEMBER(x) \
      template<class R> void serialize_internal(R &r) { \
        r.operator()x; \
      } \
      template<class R> bool deserialize_internal(R &r) { \
        r.operator()x; \
        return true; \
      }
