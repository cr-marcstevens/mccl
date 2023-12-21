#ifndef MCCL_ALGORITHM_WAGNER_HPP
#define MCCL_ALGORITHM_WAGNER_HPP

#include <mccl/config/config.hpp>
#include <mccl/algorithm/decoding.hpp>
#include <mccl/algorithm/isdgeneric.hpp>
#include <mccl/tools/unordered_multimap.hpp>
#include <mccl/tools/enumerate.hpp>

#include <unordered_map>

MCCL_BEGIN_NAMESPACE

struct wagner_config_t
{
    const std::string modulename = "wagner";
    const std::string description = "Wagner configuration";
    const std::string manualstring = 
        "Wagner:\n"
        "\tParameters: p, wd\n"
        "\tAlgorithm:\n"
        "\t\tPartition rows of G into two sets.\n\t\tCreate lists L1, L2 of (p/2) rows for each set.\n\t\tFind set W_1 of collisions between L1 & L2 on floor(log2(|L1|)) bits.\n\t\tThen find collisions between elements of W_i to produce W_(i+1) on floor(log2(|L1|)) bits\n\t\tReturn W_wd.\n"
        ;

    unsigned int p = 4;
    unsigned int wd = 2;

    template<typename Container>
    void process(Container& c)
    {
        c(p, "p", 4, "subISDT parameter p");
        c(wd, "wd", 2, "Wagner parameter wd");
    }
};

// global default. modifiable.
// at construction of subISDT_stern_dumer the current global default values will be loaded
extern wagner_config_t wagner_config_default;



class subISDT_wagner
    final : public subISDT_API
{
public:
    using subISDT_API::callback_t;

    // API member function
    ~subISDT_wagner() final
    {
        cpu_prepareloop.refresh();
        cpu_loopnext.refresh();
        cpu_callback.refresh();
        if (cpu_loopnext.total() > 0)
        {
            std::cerr << "prepare : " << cpu_prepareloop.total() << std::endl;
            std::cerr << "nextloop: " << cpu_loopnext.total() - cpu_callback.total() << std::endl;
            std::cerr << "callback: " << cpu_callback.total() << std::endl;
        }
    }
    
    subISDT_wagner()
        : config(wagner_config_default), stats("Wagner")
    {
    }

    void load_config(const configmap_t& configmap) final
    {
        mccl::load_config(config, configmap);
    }
    void save_config(configmap_t& configmap) final
    {
        mccl::save_config(config, configmap);
    }

    // API member function
    void initialize(const cmat_view& _G2, size_t _G2columns, const cvec_view& _S, unsigned int w, callback_t _callback, void* _ptr) final
    {
        if (stats.cnt_initialize._counter != 0)
            stats.refresh();
        stats.cnt_initialize.inc();

        // copy initialization parameters
        G2.reset(_G2);
        //S.reset(_S);
        columns = _G2columns;
        callback = _callback;
        ptr = _ptr;
        wmax = w;

        // copy parameters from current config
        p = config.p;
        wd = config.wd;
        // set attack parameters
        p1 = p/2; p2 = p - p1;
        rows = G2.rows();
        rows1 = rows/2; rows2 = rows - rows1;

        words = (columns+63)/64;

        // check configuration
        if (p < 2)
            throw std::runtime_error("subISDT_wagner::initialize: Wagner does not support p < 2");
        if (words > 1)
            throw std::runtime_error("subISDT_wagner::initialize: Wagner does not support l > 64 (yet)");
        if ( p > 8)
            throw std::runtime_error("subISDT_wagner::initialize: Wagner does not support p > 8 (yet)");
        if (rows1 >= 65535 || rows2 >= 65535)
            throw std::runtime_error("subISDT_wagner::initialize: Wagner does not support rows1 or rows2 >= 65535");

        firstwordmask = detail::lastwordmask(columns);
        padmask = ~firstwordmask;
        
        

        // TODO: compute a reasonable reserve size
        // hashmap.reserve(...);
    }

    // API member function
    void solve() final
    {
        stats.cnt_solve.inc();
        prepare_loop();
        while (loop_next())
            ;
    }
    
    // API member function
    void prepare_loop() final
    {
        stats.cnt_prepare_loop.inc();
        MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_prepareloop);
        
        firstwords.resize(rows);
        for (unsigned i = 0; i < rows; ++i)
            firstwords[i] = (*G2.word_ptr(i)) & firstwordmask;
        //Sval = (*S.word_ptr()) & firstwordmask;
        
        //bitfield.clear();
        hashmap.clear();
    }

    // API member function
    bool loop_next() final
    {
        stats.cnt_loop_next.inc();
        MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_loopnext);

        // stage 1: store left-table in bitfield
        /*
        enumerate.enumerate_val(firstwords.data()+rows2, firstwords.data()+rows, p1,
            [this](uint64_t val)
            { 
                bitfield.stage1(val); 
            });
        */
        // stage 2: compare right-table with bitfield: store matches
        // note we keep the packed indices at offset 0 in firstwords for right-table
        enumerate.enumerate(firstwords.data()+0, firstwords.data()+rows2, p2,
            [this](const uint32_t* idxbegin, const uint32_t* idxend, uint64_t val)
            {
                //val ^= Sval;
                //if (bitfield.stage2(val))
                    hashmap.emplace(val, pack_indices(idxbegin,idxend) );
            });
        // stage 3: retrieve matches from left-table and process
        enumerate.enumerate(firstwords.data()+rows2, firstwords.data()+rows, p1,
            [this](const uint32_t* idxbegin, const uint32_t* idxend, uint64_t val)
            {
                //if (bitfield.stage3(val))
                {
                    uint32_t* it = idx+0;
                    // note that left-table indices are offset rows2 in firstwords
                    for (auto it2 = idxbegin; it2 != idxend; ++it2,++it)
                        *it = *it2 + rows2;
                    auto range = hashmap.equal_range(val);
                    for (auto valit = range.first; valit != range.second; ++valit)
                    {
                        if (valit->first != val)
                            throw;
                        uint64_t packed_indices = valit->second;
                        auto it2 = unpack_indices(packed_indices, it);

                        MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_callback);
                        if (!(*callback)(ptr, idx+0, it2, 0))
                            return false;
                    }
                }
                return true;
            });
        return false;
    }
    
    static uint64_t pack_indices(const uint32_t* begin, const uint32_t* end)
    {
        uint64_t x = ~uint64_t(0);
        for (; begin != end; ++begin)
        {
            x <<= 16;
            x |= uint64_t(*begin);
        }
        return x;
    }
    
    uint32_t* unpack_indices(uint64_t x, uint32_t* first)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            uint32_t y = uint32_t(x & 0xFFFF);
            if (y == 0xFFFF)
                break;
            *first = y;
            ++first;
            x >>= 16;
        }
        return first;
    }

    decoding_statistics get_stats() const { return stats; };

private:
    callback_t callback;
    void* ptr;
    cmat_view G2;
    //cvec_view S;
    size_t columns, words;
    unsigned int wmax;
    
    std::unordered_multimap<uint64_t, uint64_t> hashmap;
    
    enumerate_t<uint32_t> enumerate;
    uint32_t idx[16];

    std::vector<uint64_t> firstwords;
    uint64_t firstwordmask, padmask, Sval;
    
    size_t p, p1, p2, wd, rows, rows1, rows2;
    
    wagner_config_t config;
    decoding_statistics stats;
    cpucycle_statistic cpu_prepareloop, cpu_loopnext, cpu_callback;
};


/*
template<size_t _bit_alignment = 64>
using ISD_stern_dumer = ISD_generic<subISDT_stern_dumer,_bit_alignment>;

vec solve_SD_stern_dumer(const cmat_view& H, const cvec_view& S, unsigned int w);
static inline vec solve_SD_stern_dumer(const syndrome_decoding_problem& SD)
{
    return solve_SD_stern_dumer(SD.H, SD.S, SD.w);
}

vec solve_SD_stern_dumer(const cmat_view& H, const cvec_view& S, unsigned int w, const configmap_t& configmap);
static inline vec solve_SD_stern_dumer(const syndrome_decoding_problem& SD, const configmap_t& configmap)
{
    return solve_SD_stern_dumer(SD.H, SD.S, SD.w, configmap);
}
*/


MCCL_END_NAMESPACE

#endif
