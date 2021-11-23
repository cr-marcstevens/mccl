#include <mccl/config/config.hpp>

#include <mccl/tools/enumerate.hpp>
#include <mccl/tools/utils.hpp>

#include "test_utils.hpp"

#include <iostream>
#include <vector>
#include <set>
#include <utility>

using namespace mccl;

int main(int, char**)
{
    int status = 0;

    size_t k = 20; // number of columns


    std::vector<uint64_t> firstwords(k, 0);
    for(size_t i=0; i < k; i++)
      firstwords[i] |= ((uint64_t) 1) << i;


    enumerate_t<uint32_t> enumerate;

    size_t p_up_to = 4;

    // test enumerate_val
    for(size_t p = 1; p <= p_up_to; p++) {
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
      for(size_t pp = 1; pp <= p; pp++) {
        auto weight_pp = detail::binomial<bigint_t>(k,pp);
        expected_sums += weight_pp;
        status |= (weights[pp]!=weight_pp);
      }

      status |= (expected_sums!=vals.size());
    }

    // test enumerate with indices
    for(size_t p = 1; p <= p_up_to; p++) {
      std::set<uint64_t> vals;
      std::vector<size_t> weights(p+1, 0);
      enumerate.enumerate(firstwords.data(), firstwords.data()+k, p, 
        [&vals, &weights, &status, p](uint32_t* begin, uint32_t* end, uint64_t val)
        {
            size_t w = (size_t)__builtin_popcountll(val);
            status |= (w > p);
            vals.insert(val);
            if(w<=p)
              weights[w]++;

            // indices check
            size_t indices = end-begin;
            if(indices != w) {
              status |= 1;
              std::cerr << "Incorrect number of indices: " << indices << " instead of " << w << std::endl;
            }
            std::set<uint32_t> indices_set;
            for(size_t i=0; i<indices; i++) {
              uint32_t ind = *(begin+i);
              indices_set.insert(ind);
              if(!(val&(((uint64_t) 1)<<ind))) {
                status |= 1;
              }
            }
            status |= (indices_set.size()!=indices);

            return true;
        });

      bigint_t expected_sums = 0;
      for(size_t pp = 1; pp <= p; pp++) {
        auto weight_pp = detail::binomial<bigint_t>(k,pp);
        expected_sums += weight_pp;
        status |= (weights[pp]!=weight_pp);
      }

      status |= (expected_sums!=vals.size());
    }

    if (status == 0)
    {
        LOG_CERR("All tests passed.");
        return 0;
    }
    return -1;
}
