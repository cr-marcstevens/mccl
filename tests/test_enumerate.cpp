#include <mccl/config/config.hpp>
#include <mccl/contrib/program_options.hpp>

#include <mccl/tools/enumerate.hpp>
#include <mccl/tools/utils.hpp>

#include "test_utils.hpp"

#include <iostream>
#include <vector>
#include <set>
#include <utility>
#include <random>
#include <chrono>
#include <typeinfo>
#include <cassert>

using namespace mccl;
namespace po = program_options;

typedef std::conditional<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>::type bench_clock_t;

template<typename Enumerate>
int test_enumerate_val(size_t k = 64, size_t p_up_to = 4)
{
    assert(k<=64); // test only works for k<=64
    int status = 0;
    
    std::vector<uint64_t> firstwords(k, 0);
    for(size_t i=0; i < k; i++)
        firstwords[i] |= ((uint64_t) 1) << i;

    Enumerate enumerate;

    // test enumerate_val
    for(size_t p = 1; p <= p_up_to; ++p)
    {
        std::set<uint64_t> vals;
        std::vector<size_t> weights(p+1, 0);
        enumerate.enumerate_val(firstwords.data(), firstwords.data()+k, p, 
          [&vals, &weights, &status, p](uint64_t val)
          {
              size_t w = (size_t)__builtin_popcountll(val);
              status |= (w > p);
              vals.insert(val);
              if(w<=p)
                weights[w]++;
              return true;
          });

        bigint_t expected_sums = 0;
        for(size_t pp = 1; pp <= p; pp++)
        {
          auto weight_pp = detail::binomial<bigint_t>(k,pp);
          expected_sums += weight_pp;
          status |= (weights[pp]!=weight_pp);
        }

        status |= (expected_sums!=vals.size());
    }
    if (status)
      std::cerr << "Failed!" << std::endl;
    return status;
}

template<typename Enumerate>
int test_enumerate_idx(size_t k = 64, size_t p_up_to = 4)
{
    assert(k<=64); // test only works for k<=64
    int status = 0;
    
    std::vector<uint64_t> firstwords(k, 0);
    for(size_t i=0; i < k; i++)
        firstwords[i] |= ((uint64_t) 1) << i;

    Enumerate enumerate;

    // test enumerate with indices
    for(size_t p = 1; p <= p_up_to; ++p)
    {
        std::set<uint64_t> vals;
        std::vector<size_t> weights(p+1, 0);
        enumerate.enumerate(firstwords.data(), firstwords.data()+k, p, 
          [&vals, &weights, &status, p](uint32_t* begin, uint32_t* end, uint64_t val)
          {
              size_t w = (size_t)__builtin_popcountll(val);
              status |= (w > p);
              vals.insert(val);
              if (w<=p)
                weights[w]++;

              // indices check
              size_t indices = end-begin;
              if(indices != w)
              {
                status |= 1;
                std::cerr << "Incorrect number of indices: " << indices << " instead of " << w << std::endl;
              }
              std::set<uint32_t> indices_set;
              for(size_t i=0; i<indices; i++)
              {
                uint32_t ind = *(begin+i);
                indices_set.insert(ind);
                if (!(val&(((uint64_t) 1)<<ind)))
                {
                  std::cerr << "Err2" << std::endl;
                  status |= 1;
                }
              }
              status |= (indices_set.size()!=indices);

              return true;
          });

        bigint_t expected_sums = 0;
        for(size_t pp = 1; pp <= p; pp++)
        {
          auto weight_pp = detail::binomial<bigint_t>(k,pp);
          expected_sums += weight_pp;
          status |= (weights[pp]!=weight_pp);
        }

        status |= (expected_sums!=vals.size());
    }
    if (status)
      std::cerr << "Failed!" << std::endl;
    return status;
}

template<typename Enumerate>
int bench_enumerate_val(size_t k = 256, size_t p = 4, size_t trials = 10)
{
    Enumerate enumerate;

    // prepare list
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::vector<uint64_t> words(k, 0);
    for (size_t i=0; i < k; ++i)
        words[i] = mt();

    uint64_t sm = 0;
    auto f = [&sm](uint64_t val)
        {
            sm += val;
            return true;
        };

    // warmup
    enumerate.enumerate_val(words.data(), words.data()+k, p, f);    

    // test 
    auto start0 = bench_clock_t::now();
    for(size_t t=0; t<trials;++t)
        enumerate.enumerate_val(words.data(), words.data()+k, p, f);
    auto end0 = bench_clock_t::now();
    double ms0 = std::chrono::duration_cast<std::chrono::milliseconds>(end0-start0).count();
    if (sm == 0) throw;
    std::cout << "Time: " << ms0 << " ms" << std::endl;
    return 0;
}


int main(int argc, char** argv)
{
    po::options_description allopts;
    allopts.add_options()
        ("bench,b", "Benchmark unordered_multimaps")
        ("help,h",  "Show options")
        ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, allopts), vm);
    po::notify(vm);
    if (vm.count("help"))
    {
        std::cout << allopts << std::endl;
        return 0;
    }
    if(vm.count("bench"))
    {
        // run bench
        std::cout << "\n====== Benchmark enumerate" << std::endl;
        size_t k= 256;
        size_t p = 4;
        size_t trials = 10;

        std::cout << "Run enumerate enum_opt_t<uint32_t> with options k=" << k << ", p=" << p << std::endl;
        bench_enumerate_val< enum_opt_t<uint32_t> >(k,p,trials);
        std::cout << "Run enumerate enumerate_t<uint32_t> with options k=" << k << ", p=" << p << std::endl;
        bench_enumerate_val< enumerate_t<uint32_t> >(k,p,trials);
        std::cout << "Run enumerate chase_t<uint32_t> with options k=" << k << ", p=" << p << std::endl;
        bench_enumerate_val< chase_t<uint32_t> >(k,p,trials);


        return 0;
    }

    // run tests
    int status = 0;
    size_t k = 20, p_up_to = 4;

    std::cout << "Testing enumerate_t<uint32_t>" << std::endl;
    status |= test_enumerate_val< enumerate_t<uint32_t> >(k, p_up_to);
    status |= test_enumerate_idx< enumerate_t<uint32_t> >(k, p_up_to);

    std::cout << "Testing enum_opt_t<uint32_t>" << std::endl;
    status |= test_enumerate_val< enum_opt_t<uint32_t> >(k, p_up_to);
    status |= test_enumerate_idx< enumerate_t<uint32_t> >(k, p_up_to);

    std::cout << "Testing chase_t<uint32_t>" << std::endl;
    status |= test_enumerate_val< chase_t<uint32_t> >(k, p_up_to);
    status |= test_enumerate_idx< chase_t<uint32_t> >(k, p_up_to); // not implemented yet

    if (status == 0)
    {
        LOG_CERR("All tests passed.");
        return 0;
    }
    return -1;
}
