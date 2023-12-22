#ifndef MCCL_CORE_MATRIX_LWSFORM_HPP
#define MCCL_CORE_MATRIX_LWSFORM_HPP

#include <mccl/config/config.hpp>
#include <mccl/core/matrix.hpp>
#include <mccl/core/random.hpp>
#include <mccl/tools/aligned_vector.hpp>
#include <mccl/tools/utils.hpp>
#include <mccl/tools/enumerate.hpp>

#include <algorithm>
#include <numeric>
#include <fstream>

MCCL_BEGIN_NAMESPACE

template<typename Int>
std::vector<Int> remove_solution_doublerows(const std::vector<Int>& sol)
{
	std::vector<Int> ret(sol);
	std::sort(ret.begin(), ret.end());
	unsigned i = 0;
	while (i < ret.size())
	{
		if (i + 1 < ret.size() && ret[i + 1] == ret[i])
		{
			ret.erase(ret.begin() + i, ret.begin() + i + 2);
		}
		else
			++i;
	}
	return ret;
}

class wagner_search
{
public:
	typedef std::pair<uint64_t, uint64_t> pair_t;
	static const size_t _hashmap_density = 4;
	static const size_t _hashmap_prefetchcache = 128;

	wagner_search() { _key_offset = 1; }

	unsigned collision_bits(int rows, int p)
	{
		if (p <= 0 || p > 4)
			throw;
		uint64_t r = (rows + 1) / 2;
		if (log2(r) * p > 64)
			throw;
		uint64_t count = 0;
		for (int i = 1; i <= p; ++i)
			count += detail::binomial<uint64_t>(r, i);
		return detail::log2(count);
	}

	void reserve(int rows, int p)
	{
		unsigned bits = collision_bits(rows, p);
		size_t newsize = size_t(_hashmap_density) << bits;
		if (_hashmap.size() < newsize)
		{
			_hashmap.resize(newsize);
			for (unsigned i = 0; i < _hashmap.size(); ++i)
				_hashmap[i].first = i / _hashmap_density;
		}
		newsize = size_t(2) << bits;
		if (_curresults.size() < newsize)
			_curresults.resize(newsize);
	}

	void prepare(const cmat_view& G2)
	{
		_columns = G2.columns();
		if (_columns < 64)
			_firstwordmask = detail::lastwordmask(_columns);
		else
			_firstwordmask = ~uint64_t(0);
		_firstwords.resize(G2.rows());
		for (unsigned i = 0; i < G2.rows(); ++i)
			_firstwords[i] = (*G2[i].word_ptr()) & _firstwordmask;
		//std::cout << "Wagner: " << std::hex << _firstwords[0] << " " << _firstwords[1] << std::endl;
	}

	template<typename F>
	void enumerate1(int rowbeg, int rowend, F&& f)
	{
		if (rowend > uint64_t(1) << _index_bits) throw;
		uint64_t val = 0, idx_base = ((uint64_t(1) << (2 * _index_bits)) - 1) << (1 * _index_bits);
		for (int i = rowbeg; i < rowend; ++i)
			f(_firstwords[i], idx_base + uint64_t(i));
	}
	template<typename F>
	void enumerate2(int rowbeg, int rowend, F&& f)
	{
		if (rowend > uint64_t(1) << _index_bits) throw;
		uint64_t val = 0, idx_base = ((uint64_t(1) << (1 * _index_bits)) - 1) << (2 * _index_bits);
		for (int i1 = rowbeg; i1 < rowend - 1; ++i1)
		{
			uint64_t idx = idx_base + (uint64_t(i1) << _index_bits);
			for (int i2 = i1 + 1; i2 < rowend; ++i2)
				f(_firstwords[i1] ^ _firstwords[i2], idx + i2);
		}
	}
	template<typename F>
	void enumerate3(int rowbeg, int rowend, F&& f)
	{
		if (rowend > uint64_t(1) << _index_bits) throw;
		uint64_t val = 0, idx_base = 0;
		int mid = rowbeg + ((rowend - rowbeg) / 2);
		for (int i2 = rowbeg + 1; i2 < mid; ++i2)
		{
			for (int i1 = rowbeg; i1 < i2; ++i1)
			{
				uint64_t val = _firstwords[i2] ^ _firstwords[i1];
				uint64_t idx = idx_base + (uint64_t(i1) << (2 * _index_bits)) + (uint64_t(i2) << _index_bits);
				for (int i3 = i2 + 1; i3 < rowend; ++i3)
					f(val ^ _firstwords[i3], idx + i3);
			}
		}
		for (int i2 = mid; i2 < rowend - 1; ++i2)
		{
			for (int i3 = i2 + 1; i3 < rowend; ++i3)
			{
				uint64_t val = _firstwords[i2] ^ _firstwords[i3];
				uint64_t idx = idx_base + (uint64_t(i3) << (2 * _index_bits)) + (uint64_t(i2) << _index_bits);
				for (int i1 = rowbeg; i1 < i2; ++i1)
					f(val ^ _firstwords[i1], idx + i1);
			}
		}
	}

	void checkval(uint64_t val, uint64_t idx, unsigned c = 3)
	{
#if 0
		uint64_t val2 = 0;
		uint64_t idxmask = (uint64_t(1) << _index_bits) - 1;
		for (unsigned j = 0; j < c; ++j, idx >>= _index_bits)
			if ((idx & idxmask) != idxmask)
				val2 ^= _firstwords[idx & idxmask];
		if (val2 != val)
			throw;
#endif
	}

	void insert(uint64_t val, uint64_t idx)
	{
		uint64_t bucketidx = _bucketidx(val);
		// always try the first position directly
		if (0 != ((val ^ _hashmap[bucketidx].first) & _key_mask))
		{
			_hashmap[bucketidx].first = val;
			_hashmap[bucketidx].second = idx;
			return;
		}
		// otherwise loop for remaining options
		for (int j = 1; j < _hashmap_density; ++j)
		{
			if (0 != ((val ^ _hashmap[bucketidx + j].first) & _key_mask))
			{
				_hashmap[bucketidx + j].first = val;
				_hashmap[bucketidx + j].second = idx;
				return;
			}
		}
	}
	void cache_insert(uint64_t val, uint64_t idx)
	{
//		checkval(val, idx);
		uint64_t bucketidx = _bucketidx(val);
		__builtin_prefetch(&_hashmap[bucketidx].first, 1, 0);
		_insertcache.emplace_back(val, idx);
		if (_insertcache.size() >= _hashmap_prefetchcache)
			cache_insert_flush();
	}
	void cache_insert_flush()
	{
		for (unsigned i = 0; i < _insertcache.size(); ++i)
			insert(_insertcache[i].first, _insertcache[i].second);
		_insertcache.clear();
	}

	template<int step>
	void cache_match(uint64_t val, uint64_t idx)
	{
//		checkval(val, idx);
		uint64_t bucketidx = _bucketidx(val);
		__builtin_prefetch(&_hashmap[bucketidx].first, 1, 0);
		_matchcache.emplace_back(val, idx);
		if (_matchcache.size() >= _hashmap_prefetchcache)
			cache_match_flush<step>();
	}
	void match_list2match(uint64_t val, uint64_t idx)
	{
		uint64_t bucketidx = _bucketidx(val);
		for (int j = 0; j < _hashmap_density; ++j)
		{
			if (0 == ((val ^ _hashmap[bucketidx + j].first) & _key_mask))
				list2match(_hashmap[bucketidx + j], pair_t(val, idx));
			else
				break;
		}
	}
	template<int step>
	void cache_match_flush()
	{
		for (unsigned i = 0; i < _matchcache.size(); ++i)
		{
			if (step == 1)
				match_list2match(_matchcache[i].first, _matchcache[i].second);
		}
		_matchcache.clear();
	}

	void genlist1(int p)
	{
		if (p >= 1)
			enumerate1(0, _firstwords.size() / 2, [this](uint64_t v, uint64_t i) {this->insert(v, i); });
		if (p >= 2)
			enumerate2(0, _firstwords.size() / 2, [this](uint64_t v, uint64_t i) {this->insert(v, i); });
		if (p >= 3)
			enumerate3(0, _firstwords.size() / 2, [this](uint64_t v, uint64_t i) {this->insert(v, i); });
		if (p >= 4)
			throw;
		//cache_insert_flush();
	}
	inline void list2match(const pair_t& elm1, const pair_t& elm2)
	{
		uint64_t val2 = (elm1.first ^ elm2.first);
		//		if ((val2 & _key_mask) != 0)
					//throw;
		uint64_t idx2 = (elm1.second << (_index_bits * 3)) | elm2.second;
		//(elm2.second & ((uint64_t(1) << (_index_bits * 3)) - 1))
//			| (elm1.second << (_index_bits * 3));
		//checkval(val2, idx2, 6);
		_curresults[_curresults_size].first = val2;
		_curresults[_curresults_size].second = idx2;
		++_curresults_size;
	}

	void genlist2(int p)
	{
		if (p >= 1)
			enumerate1(_firstwords.size() / 2, _firstwords.size(), [this](uint64_t v, uint64_t i) {this->match_list2match(v, i); });
		if (p >= 2)
			enumerate2(_firstwords.size() / 2, _firstwords.size(), [this](uint64_t v, uint64_t i) {this->match_list2match(v, i); });
		if (p >= 3)
			enumerate3(_firstwords.size() / 2, _firstwords.size(), [this](uint64_t v, uint64_t i) {this->match_list2match(v, i); });
		if (p >= 4)
			throw;
		//cache_match_flush<1>();
	}

	template<typename Babai>
	void genlist3(Babai& babai)
	{
		uint32_t rowidx[32];
		uint64_t idxmask = (uint64_t(1) << _index_bits) - 1;
		for (size_t i = 0; i < _curresults_size; ++i)
		{
			if (_curresults[i].first & _key_mask) throw;
			uint64_t val = _curresults[i].first >>= _collision_bits, idx = _curresults[i].second;
			uint64_t bucketidx = _bucketidx(val);
			for (unsigned j = 0; j < _hashmap_density; ++j)
			{
				if (0 != ((val ^ _hashmap[bucketidx + j].first) & _key_mask))
				{
					_hashmap[bucketidx + j].first = val;
					_hashmap[bucketidx + j].second = idx;
					break;
				}
				unsigned nrrows = 0;
				uint64_t dec = idx;
				for (unsigned j = 0; j < 6; ++j, dec >>= _index_bits)
				{
					if ((dec & idxmask) < _firstwords.size())
					{
						rowidx[nrrows] = dec & idxmask;
						++nrrows;
					}
				}
				dec = _hashmap[bucketidx + j].second;
				for (unsigned j = 0; j < 6; ++j, dec >>= _index_bits)
				{
					if ((dec & idxmask) < _firstwords.size())
					{
						rowidx[nrrows] = dec & idxmask;
						++nrrows;
					}
				}
				babai(rowidx + 0, rowidx + nrrows);
			}
		}
	}

	template<typename Mat>
	const std::vector<uint32_t>& search_G2(const Mat& G2, int p, int wd, int maxw = 65536)
	{
		if (p <= 0 || p > 4 || wd <= 0)
			throw;
		if (collision_bits(G2.rows(), p) * wd > G2.columns())
			throw;

		reserve(G2.rows(), p);
		prepare(G2);
		tmp.resize(G2.columns());

		_index_bits = detail::log2(G2.rows());
		_collision_bits = collision_bits(G2.rows(), p) - 1;
		_key_mask = (uint64_t(1) << _collision_bits) - 1;

		/*
				std::cout << "Wagner: G2: " << G2.rows() << " x " << G2.columns() << std::endl << G2 << std::endl;
				std::cout << "Wagner: _index_bits = " << _index_bits << std::endl;
				std::cout << "Wagner: p=" << p << " wd=" << wd << std::endl;
				std::cout << "Wagner: _collision_bits = " << _collision_bits << std::endl;
				std::cout << "Wagner: _key_mask = 0x" << std::hex << _key_mask << std::dec << std::endl;
				*/

		_bestsol_w = G2.columns();
		for (unsigned i = 0; i < G2.rows(); ++i)
		{
			unsigned wi = G2[i].hw() + 1;
			if (wi < _bestsol_w)
			{
				_bestsol_rows.resize(1);
				_bestsol_rows[0] = i;
				_bestsol_w = wi;
			}
		}

		_hashmap_clear();
		_results_clear();

		//std::cout << "Wagner: genlist1" << std::endl;
		genlist1(p);

		//std::cout << "Wagner: genlist2" << std::endl;
		genlist2(p);

		//std::cout << "Wagner: results: " << _curresults_size << std::endl;
		uint64_t idxmask = (uint64_t(1) << _index_bits) - 1;
		//		size_t badresults = 0 , goodresults = 0;
		for (size_t i = 0; i < _curresults_size; ++i)
		{
			//			checkval(_curresults[i].first, _curresults[i].second, 6);
						//if (hammingweight(_curresults[i].first) + 1 > maxw)
			//				continue;
			uint64_t idx = _curresults[i].second;

			unsigned nrrows = 1;
			tmp.v_copy(G2[idx & idxmask]);
			//			uint64_t val = _firstwords[idx & idxmask];

			idx >>= _index_bits;
			for (unsigned j = 1; j < 6; ++j, idx >>= _index_bits)
			{
				if ((idx & idxmask) < G2.rows())
				{
					tmp.v_xor(G2[idx & idxmask]);
					//					val ^= _firstwords[idx & idxmask];
					++nrrows;
				}
			}
			//			if (val & _key_mask)
				//			++badresults;
					//	else
						//	++goodresults;

			if (tmp.hw() + nrrows < _bestsol_w)
			{
				//std::cout << "Wagner.search_G2: improved: i=" << i << ": " << _bestsol_w << " => " << tmp.hw() + nrrows << ": 0x" << _curresults[i].second << std::endl;
				_bestsol_w = tmp.hw() + nrrows;
				_bestsol_rows.clear();
				idx = _curresults[i].second;
				for (unsigned j = 0; j < 6; ++j, idx >>= _index_bits)
				{
					if ((idx & idxmask) < G2.rows())
						_bestsol_rows.emplace_back(idx & idxmask);
				}
			}
		}
		//if (badresults)
			//throw;
		return _bestsol_rows;
	}

	template<typename Babai>
	void search_Babai(const cmat_view& G2, int p, int wd, Babai& babai)
	{
		if (p <= 0 || p > 4 || wd <= 0)
			throw;
		if (collision_bits(G2.rows(), p) * wd > G2.columns())
			throw;

		reserve(G2.rows(), p);
		prepare(G2);
		tmp.resize(G2.columns());

		_index_bits = detail::log2(G2.rows());
		_collision_bits = collision_bits(G2.rows(), p) - 1;
		if (G2.columns() < _collision_bits * wd)
			throw;
		_key_mask = (uint64_t(1) << _collision_bits) - 1;

		/*
		std::cout << "Wagner: G2: " << G2.rows() << " x " << G2.columns() << std::endl << G2 << std::endl;
		std::cout << "Wagner: _index_bits = " << _index_bits << std::endl;
		std::cout << "Wagner: p=" << p << " wd=" << wd << std::endl;
		std::cout << "Wagner: _collision_bits = " << _collision_bits << std::endl;
		std::cout << "Wagner: _key_mask = 0x" << std::hex << _key_mask << std::dec << std::endl;
		*/

		_hashmap_clear();
		_results_clear();

		//std::cout << "Wagner: genlist1" << std::endl;
		genlist1(p);

		//std::cout << "Wagner: genlist2" << std::endl;
		genlist2(p);

		if (wd == 1)
		{
			uint64_t idxmask = (uint64_t(1) << _index_bits) - 1;
			uint32_t rowidx[32];
			for (size_t i = 0; i < _curresults_size; ++i)
			{
				uint64_t idx = _curresults[i].second;
				rowidx[0] = idx & idxmask;
				unsigned nrrows = 1;
				idx >>= _index_bits;
				for (unsigned j = 1; j < 6; ++j, idx >>= _index_bits)
				{
					if ((idx & idxmask) != idxmask)
					{
						rowidx[nrrows] = idx & idxmask;
						++nrrows;
					}
				}
				babai(rowidx + 0, rowidx + nrrows);
			}
			return;
		}
		if (wd == 2)
		{
			_hashmap_clear();
			genlist3(babai);
		}

	}

private:
	void _hashmap_clear()
	{
		// this automatically invalidates all currently stored values
		++_key_offset;
	}
	void _results_clear()
	{
		_curresults_size = 0;
	}
	size_t _bucketidx(uint64_t val)
	{
		return ((val ^ _key_offset) & _key_mask) * _hashmap_density;
	}

	vec tmp;

	mccl::aligned_vector< uint64_t > _firstwords;
	mccl::aligned_vector< pair_t > _hashmap, _curresults, _insertcache, _matchcache;
	size_t _curresults_size;

	std::vector<uint32_t> _bestsol_rows;
	unsigned _bestsol_w;
	uint64_t _firstwordmask;
	uint64_t _key_offset;
	uint64_t _key_mask;
	unsigned _columns;
	unsigned _index_bits, _collision_bits;
	unsigned _maxw;
};



/*
   Class to bring G in desired LWS form:
	  G = ( 0   | 0 | G1 )
		  ( G2  | I | G3 )
	  where G1 is an epipodal basis form
	  for Babai lifting of low weight vectors from (G2 | I)

   For the purpose of efficient Babai lifting, addition 0-columns can be inserted
   to align each epipodal basis support to minimize splitting over 64-bit words & over SIMD-blocks

*/
template<size_t _bits = 256, bool _masked = false>
class G_LWS_form_t
{
public:
	static const size_t bit_alignment = _bits;
	typedef block_tag<bit_alignment, _masked> this_block_tag;
	typedef block_tag<bit_alignment, true> this_block_tag_masked;

	G_LWS_form_t() {}
	G_LWS_form_t(const cmat_view& G_) { reset(G_); }

	void reset(const cmat_view& G_, int G1_rows_ = 0)
	{
		assert(G1_rows_ < G_.rows());

		G.resize(G_.rows(), G_.columns());
		G.m_copy(G_);

		P.resize(G_.rows(), G_.columns()); // ensure enough space for future
		tmpv.resize(G_.columns());

		G_columns = G_.columns();
		G_rows = G_.rows();

		// setup column perm
		perm.resize(G_columns);
		std::iota(perm.begin(), perm.end(), 0);

		std::cout << "l0=" << G[0].hw() << std::endl;

		update_G1(G1_rows_);
		//echelonize_G2I3();
		update1();
	}

	void update_G1(int G1_rows_ = 0)
	{
		Gi_len.clear();
		// bring G1 into proper form by applying permutation
		int G1b = G_columns - 1; // points to next column to extend G1 with
		for (unsigned i = 0; i < G1_rows_; ++i)
		{
			Gi_len.emplace_back(G1b + 1);
			int bs = 0; // points to next bit position to search for 1 on row i
			while (true)
			{
				// while pos G1b contains 1 we can just decrement G1b
				for (; G1b >= 0 && G(i, G1b); --G1b)
					;
				// pos G1b contains 0: search for next 1 bit to swap with
				for (; bs < G1b && !G(i, bs); ++bs)
					;
				if (bs >= G1b)
					break;
				// swap columns
				G.swapcolumns(G1b, bs);
				std::swap(perm[G1b], perm[bs]);
				--G1b;
				++bs;
			}
		}
		//assert(G1b == G_columns - G1_columns);

		// update P
		P.resize(G1_rows_, G_columns);  // actual size
		for (int i = 0; i < G1_rows_; ++i)
		{
			P[i].v_copy(G[i]);
			if (i > 0)
				P[i].v_or(P[i - 1]);
		}

		G1_rows = G1_rows_;
		G1_columns = (G1_rows > 0) ? P[G1_rows - 1].hw() : 0;

		G23_rows = G_rows - G1_rows;
		G2_columns = G_columns - G1_columns - G23_rows;
		G2I_columns = G_columns - G1_columns;

		// create views
		// mat_view_t<this_block_tag> _G01, _G2I3;
		// mat_view_t<this_block_tag_masked> _G2, _G2I;
		_G01.reset(G.submatrix(0, G1_rows));
		_G2I3.reset(G.submatrix(G1_rows, G23_rows));

		_G2.reset(G.submatrix(G1_rows, G23_rows, G2_columns));
		_G2I.reset(G.submatrix(G1_rows, G23_rows, G_columns - G1_columns));
	}

	void echelonize_G2I3()
	{
		// echelonize G2I3
		auto pivit = G.begin() + G1_rows;
		for (unsigned pivotrow = G1_rows; pivotrow < G_rows; ++pivotrow, ++pivit)
		{
			unsigned pivotcol = G2_columns + (pivotrow - G1_rows);
			unsigned i = pivotrow;
			for (; i < G_rows && !G(i, pivotcol); ++i)
				;
			if (i >= G_rows)
			{
				i = pivotrow;
				unsigned j = G2_columns;
				for (; i < G_rows; ++i)
				{
					j = 0;
					for (; j < G2_columns && !G(i, j); ++j)
						;
					if (j < G2_columns)
						break;
				}
				if (i >= G_rows || j >= G2_columns || !G(i, j))
					throw;
				G.swapcolumns(pivotcol, j);
				std::swap(perm[pivotcol], perm[j]);
			}
			if (i > pivotrow)
				pivit.v_swap(G[i]);
			auto rowit = G.begin() + G1_rows;
			for (unsigned j = G1_rows; j < G_rows; ++j, ++rowit)
				if (j != pivotrow && rowit[pivotcol])
					rowit.v_xor(pivit);
		}
	}
	bool preprocess_extend_epipodal(int p, int wd, unsigned min_G2_columns = 0)
	{
		auto sol_ = Wagner.search_G2(G2(), p, wd);
		auto sol = remove_solution_doublerows(sol_);
		// take last row in sol as dest
		unsigned dest = sol.front();
		tmpv.v_copy(G[G1_rows + sol.front()]);
		for (unsigned j = 1; j < sol.size(); ++j)
		{
			if (sol[j] > dest)
				dest = sol[j];
			tmpv.v_xor(G[G1_rows + sol[j]]);
		}
		// check if adding solution would reduce G2 columns to below min_G2_columns
		if (G2_columns < min_G2_columns + (tmpv.subvector(G2I_columns).hw()))
			return false;

		G[G1_rows + dest].v_copy(G[G1_rows]);
		G[G1_rows].v_copy(tmpv);

		update_G1(G1_rows + 1);
		echelonize_G2I3();

		return true;
	}
	void preprocess_shrink_epipodal()
	{
		if (G1_rows <= 0)
			throw;
		update_G1(G1_rows - 1);
		echelonize_G2I3();

	}
	void preprocess(unsigned min_G2_columns, int p, int wd)
	{
		std::vector<size_t> old_G2_columns;
		old_G2_columns.push_back(G2_columns);
		while (G2_columns > min_G2_columns)
		{
			//std::cout << "Preprocess p=" << p << " wd=" << wd << " G1_rows=" << G1_rows << "..." << std::endl;
			if (!preprocess_extend_epipodal(p, wd, min_G2_columns))
				break;
			old_G2_columns.push_back(G2_columns);
		}
		if (G2_columns < min_G2_columns)
		{
			for (auto& n : old_G2_columns)
				std::cout << " !" << n << std::flush;
			throw;
		}
//			preprocess_shrink_epipodal();
		//std::cout << "G1 = \n" << G01() << std::endl;
		//std::cout << "G2 = \n" << G2() << std::endl;
	}

	void insert_sol(unsigned gi, const std::vector<uint32_t> _sol)
	{
		auto sol = remove_solution_doublerows(_sol);
		if (sol.size() == 0)
			return;
		
		tmpv.v_copy(G[sol.front()]);
		for (unsigned j = 1; j < sol.size(); ++j)
			tmpv.v_xor(G[sol[j]]);

		if (tmpv.subvector(Gi_len[gi]).hw() > G[gi].subvector(Gi_len[gi]).hw())
		{
			std::cout << "insert_sol " << gi << ": err: " << tmpv.subvector(Gi_len[gi]).hw() << " > " << G[gi].subvector(Gi_len[gi]).hw() << std::endl;
			std::cout << "insert_sol " << gi << ": sol: [";
			for (auto& j : sol)
				std::cout << " " << j;
			std::cout << " ]" << std::endl;
			std::cout << "insert_sol " << gi << ": _sol: [";
			for (auto& j : _sol)
				std::cout << " " << j;
			std::cout << " ]" << std::endl;
			throw;
		}
		
		G[sol.back()].v_copy(G[gi]);
		G[gi].v_copy(tmpv);
		if (gi == 0)
		{
			std::cout << "insert_sol: hw0=" << G[gi].hw() << std::endl;
		}
		if (gi == 0)
		{
			// undo column permutation
			for (unsigned j = 0; j < perm.size(); ++j)
			{
				while (perm[j] != j)
				{
					// move column j to its correct position
					G.swapcolumns(j, perm[j]);
					std::swap(perm[j], perm[perm[j]]);
				}
			}
			// write the new matrix to file
			std::ofstream ofs("G_" + std::to_string(G[gi].hw()));
			ofs << "# g" << std::endl << G << std::endl;
		}
		update_G1(gi + 1);
		echelonize_G2I3();
	}

	const std::vector<uint32_t>& permutation() const { return perm; }
	uint32_t permutation(uint32_t x) const { return perm[x]; }

	cvec_view_it operator[](size_t r) const { return G[r]; }
	cvec_view_it operator()(size_t r) const { return G[r]; }

	const cmat_view& Gpadded() const { return G; }
	//	size_t echelonrows() const { return echelon_rows; }
	//	size_t ISDrows() const { return ISD_rows; }

	const cmat_view_t<this_block_tag>& Gfull()   const { return G; }

	const cmat_view_t<this_block_tag>& G01()   const { return _G01; }
	const cmat_view_t<this_block_tag>& G2I3() const { return _G2I3; }
	const cmat_view_t<this_block_tag_masked>& G2()  const { return _G2; }
	const cmat_view_t<this_block_tag_masked>& G2I()  const { return _G2I; }


	void update1()
	{
		for (unsigned i = 0; i < G2_columns; ++i)
		{
			unsigned j = rndgen() % G23_rows;
			G.swapcolumns(i, G2_columns + j);
			P.swapcolumns(i, G2_columns + j);
			std::swap(perm[i], perm[G2_columns + j]);
		}
		echelonize_G2I3();
	}

	/*
	// swap with random row outside echelon form and bring it back to echelon form
	void swap_echelon(size_t echelon_idx, size_t ISD_idx)
	{
		if (!MCCL_VECTOR_NO_SANITY_CHECKS && (echelon_idx >= echelon_rows || echelon_rows + ISD_idx >= perm.size()))
			throw std::runtime_error("HST_ISD_form_t::swap_echelon(): bad input index");
		// swap rows
		std::swap(perm[echelon_idx], perm[echelon_rows + ISD_idx]);
		HST[echelon_idx].v_swap(HST[echelon_rows + ISD_idx]);

		// bring HST back in echelon form
		size_t pivotcol = HT_columns - echelon_idx - 1;
		auto pivotrow = HST[echelon_idx];
		pivotrow.clearbit(pivotcol);
		auto HSTrowit = HST.begin() + echelon_start;
		for (size_t r2 = echelon_start; r2 < HST.rows(); ++r2, ++HSTrowit)
			if (HST(r2, pivotcol))
				HSTrowit.v_xor(pivotrow);
		pivotrow.v_clear();
		pivotrow.setbit(pivotcol);
	}
	// update 1 echelon row
	void update1(size_t echelon_idx)
	{
		if (echelon_idx >= echelon_rows)
			throw std::runtime_error("HST_ISD_form_t::update(): bad input index");
		// ISD row must have 1-bit in column pivotcol:
		size_t pivotcol = HT_columns - echelon_idx - 1;
		// find random row to swap with
		//   must have bit set at pivot column
		//   start at random position and then do linear search
		size_t ISD_idx = rndgen() % ISD_rows;
		for (; ISD_idx < ISD_rows && HST(echelon_rows + ISD_idx, pivotcol) == false; ++ISD_idx)
			;
		// wrap around
		if (ISD_idx >= ISD_rows) // unlikely
		{
			ISD_idx = 0;
			for (; ISD_idx < ISD_rows && HST(echelon_rows + ISD_idx, pivotcol) == false; ++ISD_idx)
				;
		}
		// oh oh if we wrap around twice
		if (ISD_idx >= ISD_rows) // unlikely
			throw std::runtime_error("HST_ISD_form_t::update1(): cannot find pivot");
		swap_echelon(echelon_idx, ISD_idx);
	}
	// update 1 echelon row
	void update1_ISDseq(size_t echelon_idx)
	{
		if (echelon_idx >= echelon_rows)
			throw std::runtime_error("HST_ISD_form_t::update(): bad input index");
		// ISD row must have 1-bit in column pivotcol:
		size_t pivotcol = HT_columns - echelon_idx - 1;
		while (true)
		{
			cur_ISD_row = (cur_ISD_row + 1) % ISD_rows;
			if (HST(echelon_rows + cur_ISD_row, pivotcol))
				break;
		}
		swap_echelon(echelon_idx, cur_ISD_row);
	}

	// update 1 echelon row, choose ISD row from next one in a maintained random permutation
	void update1_ISDperm(size_t echelon_idx)
	{
		if (echelon_idx >= echelon_rows)
			throw std::runtime_error("HST_ISD_form_t::update1_ISDsubset(): bad input index");
		size_t pivotcol = HT_columns - echelon_idx - 1;
		size_t ISD_idx = 0;
		while (true)
		{
			// if we have consumed max_update_rows from our permutation then we (lazily) refresh it
			if (cur_ISD_row >= max_update_rows)
			{
				cur_ISD_row = 0;
				rnd_ISD_row = 0;
			}
			for (ISD_idx = cur_ISD_row; ISD_idx < ISD_perm.size(); ++ISD_idx)
			{
				// create random permutation just in time
				if (ISD_idx == rnd_ISD_row)
				{
					std::swap(ISD_perm[ISD_idx], ISD_perm[ISD_idx + (rndgen() % (ISD_rows - ISD_idx))]);
					++rnd_ISD_row;
				}
				if (HST(echelon_rows + ISD_perm[ISD_idx], pivotcol) == true)
					break;
			}
			if (ISD_idx < ISD_perm.size())
			{
				break;
			}
			// force new permutation
			cur_ISD_row = ISD_rows;
		}
		// move chosen index to cur_ISD_row position and do update
		std::swap(ISD_perm[cur_ISD_row], ISD_perm[ISD_idx]);
		ISD_idx = ISD_perm[cur_ISD_row];
		++cur_ISD_row;
		swap_echelon(echelon_idx, ISD_idx);
	}



	// Type 1: u times: pick a random echelon row & random ISD row to swap
	void update_type1(size_t rows)
	{
		for (size_t i = 0; i < rows; ++i)
			update1(rndgen() % echelon_rows);
	}
	// Type 2: pick u random distinct echelon rows & u random (non-distinct) ISD rows to swap
	void update_type2(size_t rows)
	{
		for (size_t i = 0; i < rows; ++i)
			std::swap(echelon_perm[i], echelon_perm[rndgen() % echelon_rows]);
		for (size_t i = 0; i < rows; ++i)
			update1(echelon_perm[i]);
	}
	// Type 3: pick u random distinct echelon rows & ISD rows to swap
	void update_type3(size_t rows)
	{
		// trigger refresh of ISD_perm
		cur_ISD_row = ISD_rows;
		// refresh echelon_perm
		for (size_t i = 0; i < rows; ++i)
			std::swap(echelon_perm[i], echelon_perm[rndgen() % echelon_rows]);
		for (size_t i = 0; i < rows; ++i)
			update1_ISDperm(echelon_perm[i]);
	}
	// Type 4: pick max_update_rows = k*(n-k)/n random distinct echelon rows & ISD rows to swap
	//         process u of them this round, keep the rest for next rounds until empty, repeat
	void update_type4(size_t rows)
	{
		for (size_t i = 0; i < rows; ++i)
		{
			// refresh echelon_perm when max_update_rows have been consumed
			if (cur_echelon_row >= max_update_rows)
			{
				for (size_t i = 0; i < max_update_rows; ++i)
					std::swap(echelon_perm[i], echelon_perm[rndgen() % echelon_rows]);
				cur_echelon_row = 0;
			}
			update1_ISDperm(echelon_perm[cur_echelon_row]);
			++cur_echelon_row;
		}
	}

	// Type 10: pick u round-robin echelon rows & round-robin scan of ISD rows
	void update_type10(size_t rows)
	{
		for (size_t i = 0; i < rows; ++i)
		{
			update1_ISDseq(cur_echelon_row);
			cur_echelon_row = (cur_echelon_row + 1) % echelon_rows;
		}
	}
	// Type 12: pick u round-robin echelon rows & u random (non-distinct) ISD rows to swap
	void update_type12(size_t rows)
	{
		for (size_t i = 0; i < rows; ++i)
		{
			update1(cur_echelon_row);
			cur_echelon_row = (cur_echelon_row + 1) % echelon_rows;
		}
	}
	// Type 13: pick u round-robin echelon rows & u random distinct ISD rows to swap
	void update_type13(size_t rows)
	{
		// trigger refresh of ISD_perm
		cur_ISD_row = ISD_rows;
		for (size_t i = 0; i < rows; ++i)
		{
			update1_ISDperm(cur_echelon_row);
			cur_echelon_row = (cur_echelon_row + 1) % echelon_rows;
		}
	}
	// Type 14: pick max_update_rows = k*(n-k)/n random distinct ISD rows to swap
	//         process u of them this round with u round-robin echelon rows
	//         keep the rest for next rounds until empty, repeat
	void update_type14(size_t rows)
	{
		for (size_t i = 0; i < rows; ++i)
		{
			update1_ISDperm(cur_echelon_row);
			cur_echelon_row = (cur_echelon_row + 1) % echelon_rows;
		}
	}

	// default choice is to use type 14
	void update(int r = -1, int updatetype = 14)
	{
		size_t rows = r > 0 ? std::min<size_t>(r, max_update_rows) : max_update_rows;
		switch (updatetype)
		{
		case 1:
			update_type1(rows);
			break;
		case 2:
			update_type2(rows);
			break;
		case 3:
			update_type3(rows);
			break;
		case 4:
			update_type4(rows);
			break;
		case 10:
			update_type10(rows);
			break;
		case 12:
			update_type12(rows);
			break;
		case 13:
			update_type13(rows);
			break;
		case 14:
			update_type14(rows);
			break;
		default:
			throw std::runtime_error("HST_ISD_form_t::update(): unknown update type");
		}
	}*/

private:
	/*
		Class to bring G in desired LWS form :
		G = (0  | 0 | G1)
			(G2 | I | G3)
		where G1 is an epipodal basis form
		for Babai lifting of low weight vectors from(G2 | I)
	*/

	mat_t<this_block_tag> G;
	mat_t<this_block_tag> P;
	vec_t<this_block_tag> tmpv;

	mat_view_t<this_block_tag> _G01, _G2I3;

	mat_view_t<this_block_tag_masked> _G2, _G2I;

	std::vector<uint32_t> perm;
	size_t G_columns, G_rows, G1_rows, G23_rows, G1_columns, G2_columns, G2I_columns;
	std::vector<uint32_t> echelon_perm;

	std::vector<size_t> Gi_len;

	wagner_search Wagner;

	mccl_base_random_generator rndgen;
};

template<size_t _bits = 256, bool _masked = false>
class Babai
{
public:
	static const size_t bit_alignment = _bits;
	typedef block_tag<bit_alignment, _masked> this_block_tag;
	typedef block_tag<bit_alignment, true> this_block_tag_masked;

	void initialize(const cmat_view_t<this_block_tag>& G, unsigned G1_rows)
	{
		_G.reset(G);
		_G01.reset(G.submatrix(0, G1_rows));
		_G2I3.reset(G.submatrix(G1_rows, G.rows() - G1_rows));
		tmp.resize(G.columns());

		_firstwords.resize(_G2I3.rows());
		for (unsigned i = 0; i < _G2I3.rows(); ++i)
			_firstwords[i] = (*_G2I3[i].word_ptr());
		//		std::cout << "Babai : " << std::hex << _firstwords[0] << " " << _firstwords[1] << std::endl;


				//std::cout << "Babai initialize: " << _G.rows() << " " << _G01.rows() << " " << _G2I3.rows() << std::endl;

		_babaisteps.resize(G1_rows);

		unsigned b = G.columns();
		for (unsigned i = 0; i < G1_rows; ++i)
		{
			unsigned bend = b;
			unsigned bbeg = b - 1;
			while (bbeg > 0 && G(i, bbeg - 1))
				--bbeg;
			_babaisteps[i].ghw = bend - bbeg;
			_babaisteps[i].gbound = (bend - bbeg) / 2 + 1;
			b = bbeg;

			// round down to 64 bit border
			bbeg -= bbeg & 63;
			_babaisteps[i].g.reset(G[i]);
			_babaisteps[i].gwindow.reset(G.subvector(i, bbeg, bend - bbeg));
			_babaisteps[i].wbeg = bbeg;
			_babaisteps[i].wlen = bend - bbeg;
			_babaisteps[i].wend = bend;

			_babaisteps[i].bestw = _babaisteps[i].ghw;
			_babaisteps[i].bestsol.clear();
			//std::cout << "G" << i << " = " << G[i] << std::endl;
			//std::cout << "li=" << _babaisteps[i].ghw << " " << _babaisteps[i].gwindow.hw() << std::endl;
			//std::cout << _babaisteps[i].gwindow << std::endl;

		}
		_G2I_columns = b;
	}

	void operator()(const uint32_t* rbeg, const uint32_t* rend)
	{
		if (rbeg == rend)
			throw;
		// XOR the chosen rows
		const uint32_t* rit = rbeg;
		tmp.v_copy(_G2I3[*rit]); ++rit;
		for (; rit != rend; ++rit)
			tmp.v_xor(_G2I3[*rit]);
//		if (tmp.subvector(44).hw())
//		{
//			std::cout << "Babai: tmp=" << tmp.subvector(44) << std::endl;
//			throw;
//		}
		//throw std::runtime_error("");
		// now babai lift it for each g*_l, ..., g*_1
		unsigned w = tmp.subvector(_G2I_columns).hw();
		uint64_t G1solpart = 0;
		for (int i = _babaisteps.size() - 1; w < _babaisteps[0].bestw && i >= 0; --i)
		{
			auto& BS = _babaisteps[i];
			unsigned gihw = detail::v_hw_and(BS.gwindow.ptr(), tmp.subvector(BS.wbeg, BS.wlen).ptr());
			if (BS.ghw - gihw < gihw)
			{
				gihw = BS.ghw - gihw;
				tmp.v_xor(BS.g);
				G1solpart |= uint64_t(1) << i;
			}
			w += gihw;
/*			if (tmp.subvector(BS.wend).hw() != w)
			{
				std::cout << w << " != " << tmp.subvector(BS.wend).hw() << std::endl;
				std::cout << tmp << std::endl;
				throw;
			}*/
			if (w < BS.bestw)
			{
				BS.bestw = w;
				BS.bestsol.clear();
				BS.bestsol.assign(rbeg, rend);
				for (unsigned j = 0; j < BS.bestsol.size(); ++j)
					BS.bestsol[j] += _babaisteps.size();
				for (unsigned j = 0; j < _babaisteps.size(); ++j)
					if ((G1solpart >> j) & 1)
						BS.bestsol.push_back(j);
			}
		}
	}
	unsigned bestw(unsigned i = 0) const { return _babaisteps[i].bestw; }
	const std::vector<uint32_t>& bestsol(unsigned i = 0) const { return _babaisteps[i].bestsol; }
	unsigned li(unsigned i = 0) const { return _babaisteps[i].ghw; }

private:
	struct babai_step
	{
		cvec_view_t<this_block_tag> g;
		cvec_view gwindow;
		unsigned wbeg, wlen, wend;
		unsigned ghw, gbound;

		std::vector<uint32_t> bestsol;
		unsigned bestw;
	};
	std::vector<babai_step> _babaisteps;

	cmat_view_t<this_block_tag> _G, _G01, _G2I3;
	vec_t<this_block_tag> tmp;
	unsigned _G2I_columns;
	std::vector<uint64_t> _firstwords;
};

MCCL_END_NAMESPACE

#endif
