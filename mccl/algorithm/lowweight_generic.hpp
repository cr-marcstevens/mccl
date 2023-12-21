// generic decoding API, virtual class from which decoding algorithms can be derived

#ifndef MCCL_ALGORITHM_LOWWEIGHT_GENERIC_HPP
#define MCCL_ALGORITHM_LOWWEIGHT_GENERIC_HPP

#ifdef _MSC_VER
#include <intrin.h>
template<typename T>
void __builtin_prefetch(T* p, int i = 0, int j = 0)
{
    _mm_prefetch(reinterpret_cast<const char*>(p), 5);
}

#endif

#include <mccl/config/config.hpp>
#include <mccl/algorithm/decoding.hpp>
#include <mccl/core/matrix_lwsform.hpp>
#include <mccl/tools/statistics.hpp>

MCCL_BEGIN_NAMESPACE

class lowweight_search_API;

struct lowweight_search_problem
{
    mat G;
    unsigned int w;

    bool check_solution(const cvec_view& c) const;

    template<typename LWS_t>
    vec solve(LWS_t& LWS);
};

bool check_LWS_solution(const cmat_view& G, unsigned int w, const cvec_view& c);


// virtual base class: interface to find a single solution for syndrome decoding
class lowweight_search_API
{
public:
    // virtual destructor, so we can properly delete a derived class through its base class pointer
    virtual ~lowweight_search_API() {}

    // load/save configuration from/to configmap
    // load_config should be called before initialize
    virtual void load_config(const configmap_t& configmap) = 0;
    virtual void save_config(configmap_t& configmap) = 0;

    // deterministic initialization for given parity check matrix H and target syndrome S
    virtual void initialize(const cmat_view& G, unsigned int w) = 0;
    virtual void initialize(const lowweight_search_problem& LWSP)
    {
        initialize(LWSP.G, LWSP.w);
    }

    // probabilistic preparation of loop invariant
    // when benchmark = true: should not early abort internal loops and skip internal processing of final solution
    virtual void prepare_loop(bool benchmark = false) = 0;

    // perform one loop iteration
    // return true to continue loop (no solution has been found yet)
    virtual bool loop_next() = 0;

    // run loop until a solution is found
    virtual void solve()
    {
        prepare_loop();
        while (loop_next())
            ;
    }

    // retrieve solution if any
    virtual cvec_view get_solution() const = 0;

    // retrieve statistics
    virtual decoding_statistics get_stats() const = 0;
};

template<typename LWS_t = lowweight_search_API>
vec solve_LWS(LWS_t& LWS, const cmat_view& G, unsigned int w)
{
    LWS.initialize(G, w);
    LWS.solve();
    return vec(LWS.get_solution());
}
template<typename LWS_t = lowweight_search_API>
vec solve_LWS(LWS_t& LWS, const lowweight_search_problem& LWSP)
{
    return solve_LWS(LWS, LWSP.G, LWSP.w);
}


struct lowweight_generic_config_t
{
    const std::string modulename = "lowweight";
    const std::string description = "Low Weight Search configuration";
    const std::string manualstring =
        "Low Weight Search:\n"
        "\tInput: k x n matrix G, weight w, subISD\n"
        "\tParameters:\n"
        "\t\tpreprocessing: preprocess strategy\n"
        "\t\t\t0: Systematize\n"
        "\t\t\t1: LLL\n"
        "\t\t\t2: EpiSort\n"
        "\t\t\t3: Stern\n"
        "\t\t\t4: Wagner2\n"
        "\tAlgorithm:\n"
        "\t\tRecursively preprocess epipodal basis:\n"
        "\t\t\tUse chosen preprocess strategy\n"
        "\t\t\tAt each level the matrix is of the form:\n"
        "\t\t\tG' = ( G1 0  0 )\n"
        "\t\t\t     ( G2 G3 I ), where\n"
        "\t\t\t      G1 is the epipodal basis so far\n"
        "\t\tUntil G1 cannot be made bigger while keeping G2 with at least c columns\n"
        "\t\tThen call subISD(G23, 0, l1)\n"
        ;

    unsigned int preprocessing = 1; // 0: Systematize, 1: LLL, 2: EpiSort, 3: Stern, 4: Wagner
    unsigned int preprocess_p = 2; // Stern/Wagner: enumerate over p rows for each list
    unsigned int preprocess_wd = 1; // Wagner iterations (1 means Stern)
    unsigned int lws_p = 3;
    unsigned int lws_wd = 1;
    bool verify_solution = true;
    unsigned int initg1rows = 1;

    template<typename Container>
    void process(Container& c)
    {
        c(preprocessing, "preprocessing", 4, "Preprocess strategy: 0, 1, 2, 3, 4");
        c(preprocess_p, "preprocess_p", 2, "Preprocess Stern/Wagner: enumerate over p rows for each list");
        c(preprocess_wd, "preprocess_wd", 1, "Preprocess Stern/Wagner: wd Wagner iterations (1=Stern)");
        c(lws_p, "lws_p", 3, "LWS Stern/Wagner: enumerate over p rows for each list");
        c(lws_wd, "lws_wd", 1, "LWS Stern/Wagner: wd Wagner iterations (1=Stern)");
        c(verify_solution, "verifysolution", true, "Set verification of solutions");
        c(initg1rows, "initg1rows", 1, "Initial number of G1 rows from loaded matrix");
    }
};

// global default. modifiable.
// at construction of lowweight_generic the current global default values will be loaded
extern lowweight_generic_config_t lowweight_generic_config_default;


/* TODO UPDATE DESCRIPTION */
// implementation of ISD_single_generic that can be instantiated with any subISD
// based on common view on transposed H
// will use reverse column ordering for column reduction on Htransposed (instead of row reduction on H)
//
// HT = ( 0   RI  ) where RI is the reversed identity matrix RI with 1's on the anti-diagonal
//      ( H2T H1T )
//
// this makes it easy to include additional columns of H1T together with H2T to subISD

template<typename subISDT_t = subISDT_API, size_t _bit_alignment = 256, bool _masked = false>
class lowweight_generic
    final : public lowweight_search_API
{
public:
    typedef typename subISDT_t::callback_t callback_t;

    static const size_t bit_alignment = _bit_alignment;
    
    typedef uint64_block_t<bit_alignment>  this_block_t;
    typedef block_tag<bit_alignment,_masked> this_block_tag;

    lowweight_generic(subISDT_t& sI)
        : subISDT(&sI), config(lowweight_generic_config_default), stats("LWS-generic")
    {
        n = k = w = 0;
    }

    ~lowweight_generic()
    {
    }

    void load_config(const configmap_t& configmap)
    {
        mccl::load_config(config, configmap);
    }
    void save_config(configmap_t& configmap)
    {
        mccl::save_config(config, configmap);
    }

    // deterministic initialization for given generator matrix G
    void initialize(const cmat_view& _G, unsigned int _w)
    {
        stats.cnt_initialize.inc();

        n = _G.columns();
        k = _G.rows();
        w = _w;
        Gorg.reset(_G);
        GLWS.reset(_G, 1);
        G3.resize(n, k);
        G3.m_clear();

        sol.clear();

        solution.resize(n);
        solution.v_clear();

        C.resize(n);
        C.v_clear();

        dummyS.resize(n);
        dummyS.v_clear();

        min_G2_columns = Wagner.collision_bits(_G.rows(), config.lws_p) * config.lws_wd;
    }

    // probabilistic preparation of loop invariant
    void prepare_loop(bool _benchmark = false)
    {
        stats.cnt_prepare_loop.inc();
        benchmark = _benchmark;

        //subISDT->initialize(GLWS.G2(), GLWS.G2().columns(), dummyS, w, make_ISD_callback(*this), this);
    }

    // perform one loop iteration, return true if successful and store result in e
    bool loop_next()
    {
        stats.cnt_loop_next.inc();
        GLWS.preprocess(min_G2_columns, config.preprocess_p, config.preprocess_wd);
        babai.initialize(GLWS.Gfull(), GLWS.G01().rows());
        Wagner.search_Babai(GLWS.G2(), config.lws_p, config.lws_wd, babai);
        unsigned soli = 0;
        for (; soli < GLWS.G01().rows(); ++soli)
        {
            auto& bsol = babai.bestsol(soli);
            if (bsol.size() > 0)
            {
                std::cout << soli << " " << std::flush;
                if (soli == 0)
                {
                    std::cout << GLWS.G01()[0].hw() << " => " << babai.bestw(0) << std::endl;
                    std::cout << "Old profile: ";
                    for (unsigned j = 0; j < GLWS.G01().rows(); ++j)
                        std::cout << babai.li(j) << " ";
                    std::cout << std::endl;
                }
                GLWS.insert_sol(soli, bsol);
                break;
            }            
        }
        if (soli >= GLWS.G01().rows())
        {
            // rerandomize
            GLWS.update1();
        }
        
        return false;

        return true;
//        HST.update(u, update_type);
        // find all subISD solutions
        //subISDT->solve();
        return !sol.empty();
    }

    // run loop until a solution is found
    void solve()
    {
        stats.cnt_solve.inc();
        prepare_loop();
        while (!loop_next())
            ;
        stats.refresh();
    }

    cvec_view get_solution() const
    {
        return cvec_view(solution);
    }

    // retrieve statistics
    decoding_statistics get_stats() const
    {
        return stats;
    };



    bool check_solution()
    {
        stats.cnt_check_solution.inc();
        if (solution.columns() == 0)
            throw std::runtime_error("lowweight_generic::check_solution: no solution");
        return check_LWS_solution(Gorg, w, solution);
    }
    

    // callback function
    inline bool callback(const uint32_t* begin, const uint32_t* end, unsigned int w1partial)
    {
            stats.cnt_callback.inc();
/*
            // weight of solution consists of w2 (=end-begin) + w1partial (given) + w1rest (computed below)
            size_t wsol = w1partial + (end - begin);
            if (wsol > w)
                return true;

            wsol = end - begin;
            if (begin == end)
            {
                // case selection size 0
                auto Sptr = S_blockptr;
                auto Cptr = C_blockptr;
                for (unsigned i = 0; i < blocks_per_row; ++i,++Sptr,++Cptr)
                {
                    wsol += hammingweight( *Cptr = *Sptr );
                    if (wsol > w)
                        return true;
                }
            } else if (begin == end-1)
            {
                // case selection size 1
                auto Sptr = S_blockptr;
                auto Cptr = C_blockptr;
                auto HTrowptr = H12T_blockptr + block_stride*(*begin);
                for (unsigned i = 0; i < blocks_per_row; ++i,++Cptr,++Sptr,++HTrowptr)
                {
                    wsol += hammingweight( *Cptr = *Sptr ^ *HTrowptr );
                    if (wsol > w)
                        return true;
                }
            } else {
                // case selection size >= 2
                auto Cptr = C_blockptr;
                auto Sptr = S_blockptr;
                for (unsigned i = 0; i < blocks_per_row; ++i,++Cptr,++Sptr)
                {
                    const uint32_t* p = begin;
                    *Cptr = *Sptr ^ *(H12T_blockptr + block_stride*(*p) + i);
                    for (++p; p != end-1; ++p)
                    {
                        *Cptr = *Cptr ^ *(H12T_blockptr + block_stride*(*p) + i);
                    }
                    wsol += hammingweight( *Cptr = *Cptr ^ *(H12T_blockptr + block_stride*(*p) + i) );
                    if (wsol > w)
                        return true;
                }
            }

            // this should be a correct solution at this point
*/
            if (benchmark)
                return true;
/*
            // 3. construct full solution on echelon and ISD part
            if (wsol != (end-begin) + hammingweight(C))
                throw std::runtime_error("ISD_generic::callback: internal error 1: w1partial is not correct?");
            sol.clear();
            for (auto p = begin; p != end; ++p)
                sol.push_back(HST.permutation( HST.echelonrows() + *p ));
            for (size_t c = 0; c < HST.HT().columns(); ++c)
            {
                if (C[c] == false)
                    continue;
                if (c < HST.H2T().columns())
                    throw std::runtime_error("ISD_generic::callback: internal error 2: H2T combination non-zero!");
                sol.push_back(HST.permutation( HST.HT().columns() - 1 - c ));
            }
            solution = vec(HST.HT().rows());
            for (unsigned i = 0; i < sol.size(); ++i)
                solution.setbit(sol[i]);
            if (config.verify_solution && !check_solution())
                throw std::runtime_error("ISD_generic::callback: internal error 3: solution is incorrect!");
*/
            return false;
    }
    
    
private:
    subISDT_t* subISDT;

    wagner_search Wagner;
    unsigned min_G2_columns;

    // original generator matrix G
    cmat_view Gorg;
    // solution with respect to original G
    std::vector<uint32_t> sol;
    vec solution;

    // maintains G in LWS form
    G_LWS_form_t<_bit_alignment, _masked> GLWS;
    Babai<_bit_alignment, _masked> babai;
    mat_t<this_block_tag> G3;

    // temporary vector to compute sum of syndrome and H columns
    vec_t<this_block_tag> C;
    vec_t<this_block_tag> dummyS;
    
    // block pointers to H12T, S and C
    size_t block_stride, blocks_per_row;
    const this_block_t* H12T_blockptr;
    const this_block_t* S_blockptr;
    this_block_t* C_blockptr;
    
    
    // parameters
    lowweight_generic_config_t config;

    size_t n, k, w;
    bool benchmark;
    
    // iteration count
    decoding_statistics stats;
};

MCCL_END_NAMESPACE

#endif
