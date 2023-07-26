#pragma once

namespace tf
{
    struct Vec2
    {
        float x;
        float y;
        Vec2() : x(0), y(0) {}
        Vec2(float _x, float _y) : x(_x), y(_y) {}
        Vec2 operator/(float scale) const
        {
            return Vec2(x / scale, y / scale);
        }
        Vec2 operator*(float scale) const
        {
            return Vec2(x * scale, y * scale);
        }
        Vec2 operator+(const Vec2 &rhs) const
        {
            return Vec2(x + rhs.x, y + rhs.y);
        }
        Vec2 operator=(const Vec2 &rhs)
        {
            x = rhs.x;
            y = rhs.y;
            return *this;
        }
    };

} // namespace tf
