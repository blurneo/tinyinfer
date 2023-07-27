#pragma once
#include <vector>

namespace tf
{
    template <int size>
    struct Vec
    {
        std::vector<float> vals;
        Vec() { vals.resize(size); }
        template <class T>
        Vec(T v)
        {
            vals.push_back(v);
        }
        template <class T, class... T2>
        Vec(T v, T2... rest)
        {
            vals.push_back(v);
            Vec(rest...);
        }
        Vec<size> operator/(float scale) const
        {
            Vec<size> ret;
            for (int i = 0; i < vals.size(); i++)
            {
                ret.vals[i] = vals[i] / scale;
            }
            return ret;
        }
        Vec<size> operator*(float scale) const
        {
            Vec<size> ret;
            for (int i = 0; i < vals.size(); i++)
            {
                ret.vals[i] = vals[i] * scale;
            }
            return ret;
        }
        Vec<size> operator+(const Vec<size> &rhs) const
        {
            Vec<size> ret;
            for (int i = 0; i < vals.size(); i++)
            {
                ret.vals[i] = vals[i] + rhs.vals[i];
            }
            return ret;
        }
        void operator=(const Vec<size> &rhs)
        {
            for (int i = 0; i < vals.size(); i++)
            {
                vals[i] = rhs.vals[i];
            }
        }
        float operator[](int idx) const
        {
            return vals[idx];
        }
        float &operator[](int idx)
        {
            return vals[idx];
        }
    };
    typedef Vec<2> Vec2;

} // namespace tf
