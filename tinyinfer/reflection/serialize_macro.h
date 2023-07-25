#pragma once

namespace ti
{

#define DEFINE_SERIALIZE_MEMBER(x) \
  template <class R>               \
  void serialize_internal(R &r)    \
  {                                \
    r.operator() x;                \
  }                                \
  template <class R>               \
  bool deserialize_internal(R &r)  \
  {                                \
    r.operator() x;                \
    return true;                   \
  }

  constexpr unsigned long NET_START_VAL = 0x9E241DA57C3BDA6F;
  constexpr unsigned long NET_END_VAL = 0xC56EF15ABCD46FAB;
  constexpr unsigned long LAYER_START_VAL = 0x2B74ACF267FEAC89;
  constexpr unsigned long LAYER_END_VAL = 0x6278DCAB235FEA29;
  constexpr unsigned long EMPTY_VAL = 0xE6A1E704A815F3;

}
