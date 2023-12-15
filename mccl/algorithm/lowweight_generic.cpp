#include <mccl/algorithm/lowweight_generic.hpp>

MCCL_BEGIN_NAMESPACE

template class lowweight_generic<subISDT_API,64>;
template class lowweight_generic<subISDT_API,128>;
template class lowweight_generic<subISDT_API,256>;
template class lowweight_generic<subISDT_API,512>;

lowweight_generic_config_t lowweight_generic_config_default;

bool check_LWS_solution(const cmat_view& G, unsigned int w, const cvec_view& E)
{
    if (E.columns() != G.columns())
        throw std::runtime_error("check_LWS_solution(): G and E do not have matching dimensions");
    // first check if weight of E is less or equal to w
    if (hammingweight(E) > w)
        return false;
    // then check if columns of H marked by E sum up to S
    return false;
/*    vec tmp(S.columns());
    for (size_t i = 0; i < H.rows(); ++i)
        if (hammingweight_and(H[i], E) % 2)
            tmp.setbit(i);
    return tmp.is_equal(S);*/
}

bool lowweight_search_problem::check_solution(const cvec_view& E) const
{
    return check_LWS_solution(G, w, E);
}

MCCL_END_NAMESPACE
