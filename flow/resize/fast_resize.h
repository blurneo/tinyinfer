#pragma once

template <typename T>
inline T FastResizeDivideUp(T n, T d)
{
    //ASSERT_INTEGER_TYPE( T );

    return (n + d - 1) / d;
}

template <typename T>
inline T FastResizeRoundUp(T n, T d)
{
    return FastResizeDivideUp(n, d) * d;
}

// srcstep source step in bytes
void resize_bilinear_c1(const unsigned char *src, int srcw, int srch, int srcstep, unsigned char *dst, int w, int h, int numThreads = 1);

// srcstep source step in bytes
void resize_bilinear_c3(const unsigned char *src, int srcw, int srch, int srcstep, unsigned char *dst, int w, int h, int numThreads = 1);

// srcstep source step in bytes
void resize_bilinear_c4(const unsigned char *src, int srcw, int srch, int srcstep, unsigned char *dst, int w, int h, int numThreads = 1);
