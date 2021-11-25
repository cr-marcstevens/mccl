#ifndef MCCL_TOOLS_ENUMERATE_HPP
#define MCCL_TOOLS_ENUMERATE_HPP

#include <mccl/config/config.hpp>

MCCL_BEGIN_NAMESPACE

template<typename Idx = uint16_t>
class enumerate_t
{
public:
    typedef Idx index_type;

    // if return type of f is void always return true (continue enumeration)
    template<typename F, typename ... Args>
    inline auto call_function(F&& f, Args&& ... args)
        -> typename std::enable_if<std::is_same<void,decltype(f(std::forward<Args>(args)...))>::value, bool>::type
    {
        f(std::forward<Args>(args)...);
        return true;
    }
    
    // if return type of f is bool return output of f (true to continue enumeration, false to stop)
    template<typename F, typename ... Args>
    inline auto call_function(F&& f, Args&& ... args)
        -> typename std::enable_if<std::is_same<bool,decltype(f(std::forward<Args>(args)...))>::value, bool>::type
    {
        return f(std::forward<Args>(args)...);
    }


    template<typename T, typename F>
    void enumerate1_val(const T* begin, const T* end, F&& f)
    {
        for (; begin != end; ++begin)
            if (!call_function(f,*begin))
                return;
    }

    template<typename T, typename F>
    void enumerate12_val(const T* begin, const T* end, F&& f)
    {
        if (end-begin < 2)
            return;
        for (; begin != end; )
        {
            auto val = *begin;
            if (!call_function(f,val))
                return;
            for (auto it = ++begin; it != end; ++it)
                if (!call_function(f,val ^ *it))
                    return;
        }
    }

    template<typename T, typename F>
    void enumerate2_val(const T* begin, const T* end, F&& f)
    {
        if (end-begin < 2)
            return;
        for (; begin != end; )
        {
            auto val = *begin;
            for (auto it = ++begin; it != end; ++it)
                if (!call_function(f,val ^ *it))
                    return;
        }
    }

    template<typename T, typename F>
    void enumerate3_val(const T* begin, const T* end, F&& f)
    {
        size_t count = end-begin;
        if (count < 3)
            return;
        auto mid = begin + (count/2);
        // try to have as large as possible inner loop
        // first half loop on 2nd value: use 3rd value in innerloop
        for (auto it2 = begin+1; it2 != mid; ++it2)
        {
            for (auto it1 = begin; it1 != it2; ++it1)
            {
                auto val = *it2 ^ *it1;
                for (auto it3 = it2+1; it3 != end; ++it3)
                    if (!call_function(f,val ^ *it3))
                        return;
            }
        }
        // second half loop on 2nd value: use 1st value in innerloop
        for (auto it2 = mid; it2 != end-1; ++it2)
        {
            for (auto it3 = it2+1; it3 != end; ++it3)
            {
                auto val = *it2 ^ *it3;
                for (auto it1 = begin; it1 != it2; ++it1)
                    if (!call_function(f,val ^ *it1))
                        return;
            }
        }
    }

    template<typename T, typename F>
    void enumerate4_val(const T* begin, const T* end, F&& f)
    {
        size_t count = end-begin;
        if (count < 4)
            return;
        auto mid = begin + std::min<size_t>(32, count/3);
        // try to have as large as possible inner loop
        // first half iteration: loop 2nd value until middle:
        // iterate on 3rd+4th value in inner loop, 1st in outer loop
        for (auto it2 = begin+1; it2 != mid; ++it2)
        {
            for (auto it1 = begin; it1 != it2; ++it1)
            {
                for (auto it3 = it2+1; it3 != end-1; ++it3)
                {
                    auto val = *it1 ^ *it2 ^ *it3;
                    for (auto it4 = it3+1; it4 != end; ++it4)
                    {
                        if (!call_function(f, val ^ *it4))
                            return;
                    }
                }
            }
        }
        // second half iteration: loop 2nd value from middle to end
        // iterate on 1st in inner loop, 3rd+4th value in outer loop
        for (auto it2 = mid; it2 != end-2; ++it2)
        {
            for (auto it3 = it2+1; it3 != end-1; ++it3)
            {
                for (auto it4 = it3+1; it4 != end; ++it4)
                {
                    auto val = *it4 ^ *it2 ^ *it3;
                    for (auto it1 = begin; it1 != it2; ++it1)
                    {
                        if (!call_function(f, val ^ *it1))
                            return;
                    }
                }
            }
        }
    }

    template<typename T, typename F, size_t p>
    void enumerate_p_val(const T* begin, const T* end, F&& f, T acc)
    {
        size_t count = end-begin;
        if (count < p)
            return;
        for(auto it = begin; it != end; ++it)
        {
            if constexpr(p == 1)
            {
                if (!call_function(f,*it ^ acc))
                    return;
            }
            else
            {
                enumerate_p_val<T, F, p-1>(it + 1, end, std::forward<F>(f), *it ^ acc);
            }
        }
    }

    template<typename T, typename F>
    void enumerate_val(const T* begin, const T* end, size_t p, F&& f)
    {
        switch (p)
        {
            default: throw std::runtime_error("enumerate::enumerate_val: only 1 <= p <= 4 supported");
            case 4:
                enumerate_p_val<T, F, 4>(begin,end,std::forward<F>(f), 0);
                __attribute__ ((fallthrough));
            case 3:
                enumerate_p_val<T, F, 3>(begin,end,std::forward<F>(f), 0);
                __attribute__ ((fallthrough));
            case 2:
                enumerate_p_val<T, F, 2>(begin,end,std::forward<F>(f), 0);
                __attribute__ ((fallthrough));
            case 1:
                enumerate_p_val<T, F, 1>(begin,end,std::forward<F>(f), 0);
        }
    }

    template<typename T, typename F>
    void enumerate1(const T* begin, const T* end, F&& f)
    {
        idx[0] = 0;
        for (; begin != end; ++begin,++idx[0])
            if (!call_function(f,idx+0,idx+1,*begin))
                return;
    }

    template<typename T, typename F>
    void enumerate12(const T* begin, const T* end, F&& f)
    {
        idx[0] = 0;
        for (; begin != end; ++idx[0])
        {
            auto val = *begin;
            if (!call_function(f,idx+0,idx+1,val))
                return;
            idx[1] = idx[0]+1;
            for (auto it = ++begin; it != end; ++it,++idx[1])
                if (!call_function(f,idx+0,idx+2,val ^ *it))
                    return;
        }
    }

    template<typename T, typename F>
    void enumerate2(const T* begin, const T* end, F&& f)
    {
        idx[0] = 0;
        for (; begin != end; ++idx[0])
        {
            auto val = *begin;
            idx[1] = idx[0]+1;
            for (auto it = ++begin; it != end; ++it,++idx[1])
                if (!call_function(f,idx+0,idx+2,val ^ *it))
                    return;
        }
    }

    template<typename T, typename F>
    void enumerate3(const T* begin, const T* end, F&& f)
    {
        size_t count = end-begin;
        auto mid = begin + (count/2);
        idx[1] = 1;
        // try to have as large as possible inner loop
        // first half loop on 2nd value: use 3rd value in innerloop
        for (auto it2 = begin+1; it2 != mid; ++it2,++idx[1])
        {
            idx[0] = 0;
            for (auto it1 = begin; it1 != it2; ++it1,++idx[0])
            {
                auto val = *it2 ^ *it1;
                idx[2] = idx[1]+1;
                for (auto it3 = it2+1; it3 != end; ++it3,++idx[2])
                    if (!call_function(f,idx+0, idx+3, val ^ *it3))
                        return;
            }
        }
        // second half loop on 2nd value: use 1st value in innerloop
        for (auto it2 = mid; it2 != end-1; ++it2,++idx[1])
        {
            idx[2] = idx[1]+1;
            for (auto it3 = it2+1; it3 != end; ++it3,++idx[2])
            {
                auto val = *it2 ^ *it3;
                idx[0] = 0;
                for (auto it1 = begin; it1 != it2; ++it1,++idx[0])
                    if (!call_function(f,idx+0, idx+3, val ^ *it1))
                        return;
            }
        }
    }

    template<typename T, typename F>
    void enumerate4(const T* begin, const T* end, F&& f)
    {
        size_t count = end-begin;
        if (count < 4)
            return;
        auto mid = begin + std::min<size_t>(32, count/3);
        idx[1] = 1;
        // try to have as large as possible inner loop
        // first half iteration: loop 2nd value until middle:
        // iterate on 3rd+4th value in inner loop, 1st in outer loop
        for (auto it2 = begin+1; it2 != mid; ++it2,++idx[1])
        {
            idx[0] = 0;
            for (auto it1 = begin; it1 != it2; ++it1,++idx[0])
            {
                idx[2] = idx[1]+1;
                for (auto it3 = it2+1; it3 != end-1; ++it3,++idx[2])
                {
                    auto val = *it1 ^ *it2 ^ *it3;
                    idx[3] = idx[2]+1;
                    for (auto it4 = it3+1; it4 != end; ++it4,++idx[3])
                    {
                        if (!call_function(f, idx+0, idx+4, val ^ *it4))
                            return;
                    }
                }
            }
        }
        // second half iteration: loop 2nd value from middle to end
        // iterate on 1st in inner loop, 3rd+4th value in outer loop
        for (auto it2 = mid; it2 != end-2; ++it2,++idx[1])
        {
            idx[2] = idx[1]+1;
            for (auto it3 = it2+1; it3 != end-1; ++it3,++idx[2])
            {
                idx[3] = idx[2]+1;
                for (auto it4 = it3+1; it4 != end; ++it4,++idx[3])
                {
                    auto val = *it4 ^ *it2 ^ *it3;
                    idx[0] = 0;
                    for (auto it1 = begin; it1 != it2; ++it1,++idx[0])
                    {
                        if (!call_function(f, idx+0, idx+4, val ^ *it1))
                            return;
                    }
                }
            }
        }
    }
    
    template<typename T, typename F, size_t p, size_t i=0>
    void enumerate_p(const T* begin, const T* end, F&& f, T acc, size_t cur_idx)
    {
        size_t count = end-begin;
        if (count < p-i)
            return;
        idx[i] = cur_idx;
        for(auto it = begin; it != end; ++it, ++idx[i])
        {
            if constexpr(i == p-1)
            {
                if (!call_function(f, idx+0, idx+p, *it ^ acc))
                    return;
            }
            else
            {
                enumerate_p<T, F, p, i+1>(it + 1, end, std::forward<F>(f), *it ^ acc, idx[i]+1);
            }
        }
    }

    template<typename T, typename F>
    void enumerate(const T* begin, const T* end, size_t p, F&& f)
    {
        switch (p)
        {
            default: throw std::runtime_error("enumerate::enumerate: only 1 <= p <= 4 supported");
            case 4:
                enumerate_p<T, F, 4>(begin,end,std::forward<F>(f), 0, 0);
                __attribute__ ((fallthrough));
            case 3:
                enumerate_p<T, F, 3>(begin,end,std::forward<F>(f), 0, 0);
                __attribute__ ((fallthrough));
            case 2:
                enumerate_p<T, F, 2>(begin,end,std::forward<F>(f), 0, 0);
                __attribute__ ((fallthrough));
            case 1:
                enumerate_p<T, F, 1>(begin,end,std::forward<F>(f), 0, 0);
        }
    }

    index_type idx[16];
};

template<typename T, typename F>
std::vector<std::vector<T>> precompute(const T* begin, const T* end) {
    std::vector<std::vector<T>> precomputed(2, std::vector<T> (end - begin));
    for(auto it = begin; it != end-1; ++it) {
        precomputed[0][it - begin] = *it ^ *(it+1);
    }
    for(auto it = begin; it != end-2; ++it) {
        precomputed[1][it - begin] = *it ^ *(it+2);
    }
    return precomputed;
}

template<typename Idx = uint16_t>
class chase_t
{
public:
    typedef Idx index_type;

    // if return type of f is void always return true (continue enumeration)
    template<typename F, typename... Args>
    inline auto call_function(F&& f, Args&&... args)
        -> typename std::enable_if<std::is_same<void, decltype(f(std::forward<Args>(args)...))>::value, bool>::type
    {
        f(std::forward<Args>(args)...);
        return true;
    }

    // if return type of f is bool return output of f (true to continue enumeration, false to stop)
    template<typename F, typename... Args>
    inline auto call_function(F&& f, Args&&... args)
        -> typename std::enable_if<std::is_same<bool, decltype(f(std::forward<Args>(args)...))>::value, bool>::type
    {
        return f(std::forward<Args>(args)...);
    }

    /*
     * List all combinations of 't' elements of a set of 'n' elements.
     *
     * Generate a Chase's sequence: the binary representation of a combination and
     * its successor only differ by two bits that are either consecutive of
     * separated by only one position.
     *
     * See exercise 45 of Knuth's The art of computer programming volume 4A.
     */
    template<typename T, typename F, size_t p>
    void enumerate_p_val(const T* begin, const T* end, F&& f)
    {
        std::vector<std::vector<T>> precomputed = precompute<T, F>(begin, end);
        index_type z[16];

        index_type diff_pos = 0;
        index_type diff_len = 0;
        int32_t x;
        for (size_t j = 1; j <= p + 1; ++j) {
            z[j] = 0;
        }
        auto val = 0;
        for (size_t j = 1; j <= p + 1; ++j) {
            val ^= *(end - p - 1 + j);
            idx[j] = end - begin - p - 1 + j;
        }
        if (!call_function(f, val))
            return;
        /* r is the least subscript with idx[r] >= r. */
        size_t r = 1;
        size_t j = r;

        goto novisit;
        while (1) {
            // for (size_t i = 1; i <= p; ++i) {
            //     combinations[i - 1 + N * p] = idx[i];
            // }
            val ^= precomputed[diff_len - 1][diff_pos];
            if (!call_function(f, val))
                return;
            j = r;

        novisit:
            if (z[j]) {
                x = idx[j] + 2;
                if (x < z[j]) {
                    diff_pos = idx[j];
                    diff_len = 2;
                    idx[j] = x;
                } else if (x == z[j] && z[j + 1]) {
                    diff_pos = idx[j];
                    diff_len = 2 - (idx[j + 1] % 2);
                    idx[j] = x - (idx[j + 1] % 2);
                } else {
                    z[j] = 0;
                    ++j;
                    if (j <= p)
                        goto novisit;
                    else
                        return;
                }
                if (idx[1] > 0)
                    r = 1;
                else
                    r = j - 1;
            } else {
                x = idx[j] + (idx[j] % 2) - 2;
                if (x >= (int32_t)j) {
                    diff_pos = x;
                    diff_len = 2 - (idx[j] % 2);
                    idx[j] = x;
                    r = 1;
                } else if (idx[j] == j) {
                    diff_pos = j - 1;
                    diff_len = 1;
                    idx[j] = j - 1;
                    z[j] = idx[j + 1] - ((idx[j + 1] + 1) % 2);
                    r = j;
                } else if (idx[j] < j) {
                    diff_pos = idx[j];
                    diff_len = j - idx[j];
                    idx[j] = j;
                    z[j] = idx[j + 1] - ((idx[j + 1] + 1) % 2);
                    r = (j > 2) ? j - 1 : 1;
                } else {
                    diff_pos = x;
                    diff_len = 2 - (idx[j] % 2);
                    idx[j] = x;
                    r = j;
                }
            }
        }
    }
    template<typename T, typename F>
    void enumerate_val(const T* begin, const T* end, size_t p, F&& f)
    {
        switch (p) {
        default:
            throw std::runtime_error("enumerate::enumerate_val: only 1 <= p <= 4 supported");
        case 4:
            enumerate_p_val<T, F, 4>(begin, end, std::forward<F>(f));
            __attribute__((fallthrough));
        case 3:
            enumerate_p_val<T, F, 3>(begin, end, std::forward<F>(f));
            __attribute__((fallthrough));
        case 2:
            enumerate_p_val<T, F, 2>(begin, end, std::forward<F>(f));
            __attribute__((fallthrough));
        case 1:
            enumerate_p_val<T, F, 1>(begin, end, std::forward<F>(f));
        }
    }

    index_type idx[16];
};
MCCL_END_NAMESPACE

#endif
