
/**
 * edax-reversi
 *
 * https://github.com/abulmo/edax-reversi
 *
 * @date 1998 - 2017
 * @author Richard Delorme
 * @version 4.4
 */

 /**
  * edax-reversi-AVX
  *
  * https://github.com/okuhara/edax-reversi-AVX
  *
  * @date 1998 - 2018
  * @author Toshihiko Okuhara
  * @version 4.4
  */

/*
 * @date 2020
 * @author Hiroki Takizawa
 */

// This source code is licensed under the
// GNU General Public License v3.0

#include<iostream>
#include<iomanip>
#include<set>
#include<vector>
#include<array>
#include<string>
#include<stack>
#include<algorithm>
#include<iterator>
#include<cassert>
#include<functional>
#include<random>
#include<cstdint>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <wmmintrin.h>
#include <immintrin.h>


typedef struct Board {

private:
	uint64_t transpose(uint64_t b) const {
		uint64_t t;

		t = (b ^ (b >> 7)) & 0x00aa00aa00aa00aaULL;
		b = b ^ t ^ (t << 7);
		t = (b ^ (b >> 14)) & 0x0000cccc0000ccccULL;
		b = b ^ t ^ (t << 14);
		t = (b ^ (b >> 28)) & 0x00000000f0f0f0f0ULL;
		b = b ^ t ^ (t << 28);

		return b;
	}
	uint64_t vertical_mirror(uint64_t b) const {
		b = ((b >> 8) & 0x00FF00FF00FF00FFULL) | ((b << 8) & 0xFF00FF00FF00FF00ULL);
		b = ((b >> 16) & 0x0000FFFF0000FFFFULL) | ((b << 16) & 0xFFFF0000FFFF0000ULL);
		b = ((b >> 32) & 0x00000000FFFFFFFFULL) | ((b << 32) & 0xFFFFFFFF00000000ULL);
		return b;
	}
	uint64_t horizontal_mirror(uint64_t b) const {
		b = ((b >> 1) & 0x5555555555555555ULL) | ((b << 1) & 0xAAAAAAAAAAAAAAAAULL);
		b = ((b >> 2) & 0x3333333333333333ULL) | ((b << 2) & 0xCCCCCCCCCCCCCCCCULL);
		b = ((b >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((b << 4) & 0xF0F0F0F0F0F0F0F0ULL);

		return b;
	}
	void board_check(const std::array<uint64_t, 2> &board) const {
		if (board[0] & board[1]) {
			std::cerr << "Two discs on the same square?" << std::endl;
			std::abort();
		}

		if (((board[0] | board[1]) & 0x0000001818000000ULL) != 0x0000001818000000ULL) {
			std::cerr << "Empty center?" << std::endl;
			std::abort();
		}
	}
	void board_symmetry(const int s, std::array<uint64_t, 2> &sym) const {

		std::array<uint64_t, 2>board{ player,opponent };

		if (s & 1) {
			board[0] = horizontal_mirror(board[0]);
			board[1] = horizontal_mirror(board[1]);
		}
		if (s & 2) {
			board[0] = vertical_mirror(board[0]);
			board[1] = vertical_mirror(board[1]);
		}
		if (s & 4) {
			board[0] = transpose(board[0]);
			board[1] = transpose(board[1]);
		}

		sym = board;

		board_check(sym);
	}
public:

	uint64_t player, opponent;
	Board(uint64_t p, uint64_t o) :player(p), opponent(o) {}
	Board() :player(0), opponent(0) {}
	bool operator <(const Board &b) const {
		if (this->player < b.player)return true;
		if (this->player > b.player)return false;
		if (this->opponent < b.opponent)return true;
		return false;
	}
	bool operator >(const Board &b) const {
		return b < *this;
	}
	bool operator <=(const Board &b) const {
		return !(*this > b);
	}
	bool operator >=(const Board &b) const {
		return !(*this < b);
	}
	int64_t popcount() const {
		return _mm_popcnt_u64(player) + _mm_popcnt_u64(opponent);
	}
	std::array<uint64_t,2> unique() const {

		std::array<uint64_t, 2> tmp{ 0,0 }, answer{ player, opponent };

		for (int i = 1; i < 8; ++i) {
			board_symmetry(i, tmp);
			if (tmp < answer) {
				answer = tmp;
			}
		}

		board_check(answer);
		return answer;
	}
	static Board initial() {
		return Board(0x0000000810000000ULL, 0x0000001008000000ULL);
	}

} Board;

alignas(64) const uint64_t lmask_v4[64][4] = {
	{ 0x00000000000000fe, 0x0101010101010100, 0x8040201008040200, 0x0000000000000000 },
	{ 0x00000000000000fc, 0x0202020202020200, 0x0080402010080400, 0x0000000000000100 },
	{ 0x00000000000000f8, 0x0404040404040400, 0x0000804020100800, 0x0000000000010200 },
	{ 0x00000000000000f0, 0x0808080808080800, 0x0000008040201000, 0x0000000001020400 },
	{ 0x00000000000000e0, 0x1010101010101000, 0x0000000080402000, 0x0000000102040800 },
	{ 0x00000000000000c0, 0x2020202020202000, 0x0000000000804000, 0x0000010204081000 },
	{ 0x0000000000000080, 0x4040404040404000, 0x0000000000008000, 0x0001020408102000 },
	{ 0x0000000000000000, 0x8080808080808000, 0x0000000000000000, 0x0102040810204000 },
	{ 0x000000000000fe00, 0x0101010101010000, 0x4020100804020000, 0x0000000000000000 },
	{ 0x000000000000fc00, 0x0202020202020000, 0x8040201008040000, 0x0000000000010000 },
	{ 0x000000000000f800, 0x0404040404040000, 0x0080402010080000, 0x0000000001020000 },
	{ 0x000000000000f000, 0x0808080808080000, 0x0000804020100000, 0x0000000102040000 },
	{ 0x000000000000e000, 0x1010101010100000, 0x0000008040200000, 0x0000010204080000 },
	{ 0x000000000000c000, 0x2020202020200000, 0x0000000080400000, 0x0001020408100000 },
	{ 0x0000000000008000, 0x4040404040400000, 0x0000000000800000, 0x0102040810200000 },
	{ 0x0000000000000000, 0x8080808080800000, 0x0000000000000000, 0x0204081020400000 },
	{ 0x0000000000fe0000, 0x0101010101000000, 0x2010080402000000, 0x0000000000000000 },
	{ 0x0000000000fc0000, 0x0202020202000000, 0x4020100804000000, 0x0000000001000000 },
	{ 0x0000000000f80000, 0x0404040404000000, 0x8040201008000000, 0x0000000102000000 },
	{ 0x0000000000f00000, 0x0808080808000000, 0x0080402010000000, 0x0000010204000000 },
	{ 0x0000000000e00000, 0x1010101010000000, 0x0000804020000000, 0x0001020408000000 },
	{ 0x0000000000c00000, 0x2020202020000000, 0x0000008040000000, 0x0102040810000000 },
	{ 0x0000000000800000, 0x4040404040000000, 0x0000000080000000, 0x0204081020000000 },
	{ 0x0000000000000000, 0x8080808080000000, 0x0000000000000000, 0x0408102040000000 },
	{ 0x00000000fe000000, 0x0101010100000000, 0x1008040200000000, 0x0000000000000000 },
	{ 0x00000000fc000000, 0x0202020200000000, 0x2010080400000000, 0x0000000100000000 },
	{ 0x00000000f8000000, 0x0404040400000000, 0x4020100800000000, 0x0000010200000000 },
	{ 0x00000000f0000000, 0x0808080800000000, 0x8040201000000000, 0x0001020400000000 },
	{ 0x00000000e0000000, 0x1010101000000000, 0x0080402000000000, 0x0102040800000000 },
	{ 0x00000000c0000000, 0x2020202000000000, 0x0000804000000000, 0x0204081000000000 },
	{ 0x0000000080000000, 0x4040404000000000, 0x0000008000000000, 0x0408102000000000 },
	{ 0x0000000000000000, 0x8080808000000000, 0x0000000000000000, 0x0810204000000000 },
	{ 0x000000fe00000000, 0x0101010000000000, 0x0804020000000000, 0x0000000000000000 },
	{ 0x000000fc00000000, 0x0202020000000000, 0x1008040000000000, 0x0000010000000000 },
	{ 0x000000f800000000, 0x0404040000000000, 0x2010080000000000, 0x0001020000000000 },
	{ 0x000000f000000000, 0x0808080000000000, 0x4020100000000000, 0x0102040000000000 },
	{ 0x000000e000000000, 0x1010100000000000, 0x8040200000000000, 0x0204080000000000 },
	{ 0x000000c000000000, 0x2020200000000000, 0x0080400000000000, 0x0408100000000000 },
	{ 0x0000008000000000, 0x4040400000000000, 0x0000800000000000, 0x0810200000000000 },
	{ 0x0000000000000000, 0x8080800000000000, 0x0000000000000000, 0x1020400000000000 },
	{ 0x0000fe0000000000, 0x0101000000000000, 0x0402000000000000, 0x0000000000000000 },
	{ 0x0000fc0000000000, 0x0202000000000000, 0x0804000000000000, 0x0001000000000000 },
	{ 0x0000f80000000000, 0x0404000000000000, 0x1008000000000000, 0x0102000000000000 },
	{ 0x0000f00000000000, 0x0808000000000000, 0x2010000000000000, 0x0204000000000000 },
	{ 0x0000e00000000000, 0x1010000000000000, 0x4020000000000000, 0x0408000000000000 },
	{ 0x0000c00000000000, 0x2020000000000000, 0x8040000000000000, 0x0810000000000000 },
	{ 0x0000800000000000, 0x4040000000000000, 0x0080000000000000, 0x1020000000000000 },
	{ 0x0000000000000000, 0x8080000000000000, 0x0000000000000000, 0x2040000000000000 },
	{ 0x00fe000000000000, 0x0100000000000000, 0x0200000000000000, 0x0000000000000000 },
	{ 0x00fc000000000000, 0x0200000000000000, 0x0400000000000000, 0x0100000000000000 },
	{ 0x00f8000000000000, 0x0400000000000000, 0x0800000000000000, 0x0200000000000000 },
	{ 0x00f0000000000000, 0x0800000000000000, 0x1000000000000000, 0x0400000000000000 },
	{ 0x00e0000000000000, 0x1000000000000000, 0x2000000000000000, 0x0800000000000000 },
	{ 0x00c0000000000000, 0x2000000000000000, 0x4000000000000000, 0x1000000000000000 },
	{ 0x0080000000000000, 0x4000000000000000, 0x8000000000000000, 0x2000000000000000 },
	{ 0x0000000000000000, 0x8000000000000000, 0x0000000000000000, 0x4000000000000000 },
	{ 0xfe00000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
	{ 0xfc00000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
	{ 0xf800000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
	{ 0xf000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
	{ 0xe000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
	{ 0xc000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
	{ 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
	{ 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }
};

uint64_t flip(uint64_t pos, uint64_t player, uint64_t opponent)
{
	__m256i	flip, outflank, ocontig;
	const __m256i shift1897 = _mm256_set_epi64x(7, 9, 8, 1);
	const __m256i shift1897_2 = _mm256_set_epi64x(14, 18, 16, 2);
	const __m256i mask_flip1897 = _mm256_set_epi64x(0x007e7e7e7e7e7e00, 0x007e7e7e7e7e7e00, 0x00ffffffffffff00, 0x7e7e7e7e7e7e7e7e);

	const __m256i PPPP = _mm256_set1_epi64x(player);
	const __m256i OOOO_masked = _mm256_and_si256(_mm256_set1_epi64x(opponent), mask_flip1897);
	const __m256i pre = _mm256_and_si256(OOOO_masked, _mm256_srlv_epi64(OOOO_masked, shift1897));
	const __m256i mask = _mm256_loadu_si256((__m256i*)lmask_v4[pos]);

	ocontig = _mm256_set1_epi64x(1ULL << pos);
	ocontig = _mm256_and_si256(OOOO_masked, _mm256_srlv_epi64(ocontig, shift1897));
	ocontig = _mm256_or_si256(ocontig, _mm256_and_si256(OOOO_masked, _mm256_srlv_epi64(ocontig, shift1897)));
	ocontig = _mm256_or_si256(ocontig, _mm256_and_si256(pre, _mm256_srlv_epi64(ocontig, shift1897_2)));
	ocontig = _mm256_or_si256(ocontig, _mm256_and_si256(pre, _mm256_srlv_epi64(ocontig, shift1897_2)));
	outflank = _mm256_and_si256(_mm256_srlv_epi64(ocontig, shift1897), PPPP);
	flip = _mm256_andnot_si256(_mm256_cmpeq_epi64(outflank, _mm256_setzero_si256()), ocontig);

	ocontig = _mm256_andnot_si256(OOOO_masked, mask);
	ocontig = _mm256_and_si256(ocontig, _mm256_sub_epi64(_mm256_setzero_si256(), ocontig));	// LS1B
	outflank = _mm256_and_si256(ocontig, PPPP);
	flip = _mm256_or_si256(flip, _mm256_and_si256(mask, _mm256_add_epi64(outflank, _mm256_cmpeq_epi64(outflank, ocontig))));

	const __m128i flip2 = _mm_or_si128(_mm256_castsi256_si128(flip), _mm256_extracti128_si256(flip, 1));
	return uint64_t(_mm_cvtsi128_si64(flip2) | _mm_extract_epi64(flip2, 1));
}

uint64_t get_moves(const uint64_t P, const uint64_t O)
{
	__m256i	PP, mOO, MM, flip_l, flip_r, pre_l, pre_r, shift2;
	__m128i	M;
	const __m256i shift1897 = _mm256_set_epi64x(7, 9, 8, 1);
	const __m256i mflipH = _mm256_set_epi64x(0x7e7e7e7e7e7e7e7e, 0x7e7e7e7e7e7e7e7e, -1, 0x7e7e7e7e7e7e7e7e);

	PP = _mm256_broadcastq_epi64(_mm_cvtsi64_si128(P));
	mOO = _mm256_and_si256(_mm256_broadcastq_epi64(_mm_cvtsi64_si128(O)), mflipH);

	flip_l = _mm256_and_si256(mOO, _mm256_sllv_epi64(PP, shift1897));
	flip_r = _mm256_and_si256(mOO, _mm256_srlv_epi64(PP, shift1897));
	flip_l = _mm256_or_si256(flip_l, _mm256_and_si256(mOO, _mm256_sllv_epi64(flip_l, shift1897)));
	flip_r = _mm256_or_si256(flip_r, _mm256_and_si256(mOO, _mm256_srlv_epi64(flip_r, shift1897)));
	pre_l = _mm256_and_si256(mOO, _mm256_sllv_epi64(mOO, shift1897));
	pre_r = _mm256_srlv_epi64(pre_l, shift1897);
	shift2 = _mm256_add_epi64(shift1897, shift1897);
	flip_l = _mm256_or_si256(flip_l, _mm256_and_si256(pre_l, _mm256_sllv_epi64(flip_l, shift2)));
	flip_r = _mm256_or_si256(flip_r, _mm256_and_si256(pre_r, _mm256_srlv_epi64(flip_r, shift2)));
	flip_l = _mm256_or_si256(flip_l, _mm256_and_si256(pre_l, _mm256_sllv_epi64(flip_l, shift2)));
	flip_r = _mm256_or_si256(flip_r, _mm256_and_si256(pre_r, _mm256_srlv_epi64(flip_r, shift2)));
	MM = _mm256_sllv_epi64(flip_l, shift1897);
	MM = _mm256_or_si256(MM, _mm256_srlv_epi64(flip_r, shift1897));

	M = _mm_or_si128(_mm256_castsi256_si128(MM), _mm256_extracti128_si256(MM, 1));
	M = _mm_or_si128(M, _mm_unpackhi_epi64(M, M));
	return _mm_cvtsi128_si64(M) & ~(P | O);	// mask with empties
}

std::set<std::array<uint64_t,2>>searched, leafnode;

int DISCS = 10;

void search(const Board &board) {

	const auto uni = board.unique();

	//盤上の石の数が閾値に達していて、かつ石を置くような合法手が存在すれば、leafnodeに登録する。
	if (board.popcount() >= DISCS) {
		if (get_moves(board.player, board.opponent)) {
			leafnode.insert(uni);
			return;
		}
		else if (get_moves(board.opponent, board.player)) {
			Board next;
			next.player = board.opponent;
			next.opponent = board.player;
			search(next);
		}
		return;
	}

	if (searched.find(uni) != searched.end())return;
	searched.insert(uni);

	Board next;
	uint64_t b = get_moves(board.player, board.opponent);

	if (b == 0) {
		if (get_moves(board.opponent, board.player)) {
			next.player = board.opponent;
			next.opponent = board.player;
			search(next);
		}
		return;
	}
	
	for (unsigned long index = 0; _BitScanForward64(&index, b); b &= b - 1) {
		const uint64_t flipped = flip(index, board.player, board.opponent);
		if (flipped == 0) continue;
		next.player = board.opponent ^ flipped;
		next.opponent = board.player ^ (flipped | (1ULL << index));
		search(next);
	}
}


//bがオセロの置かれている石のビットボードだとして、すべての石たちが8近傍で連結しているか調べる。しているならtrue、いないならfalseを返す。
bool IsConnected(const uint64_t b) {

	uint64_t mark = 0x0000'0018'1800'0000ULL, old_mark = 0;

	assert((b & mark) == mark);

	//真ん中4つの石にマークをつけたとして、マークがついている石の8近傍の石にマークをつける。変化しなくなれば終了。
	while (mark != old_mark) {
		old_mark = mark;
		uint64_t new_mark = mark;
		new_mark |= b & ((mark & 0xFEFE'FEFE'FEFE'FEFEULL) >> 1);
		new_mark |= b & ((mark & 0x7F7F'7F7F'7F7F'7F7FULL) << 1);
		new_mark |= b & ((mark & 0xFFFF'FFFF'FFFF'FF00ULL) >> 8);
		new_mark |= b & ((mark & 0x00FF'FFFF'FFFF'FFFFULL) << 8);
		new_mark |= b & ((mark & 0x7F7F'7F7F'7F7F'7F00ULL) >> 7);
		new_mark |= b & ((mark & 0x00FE'FEFE'FEFE'FEFEULL) << 7);
		new_mark |= b & ((mark & 0xFEFE'FEFE'FEFE'FE00ULL) >> 9);
		new_mark |= b & ((mark & 0x007F'7F7F'7F7F'7F7FULL) << 9);

		mark = new_mark;
	}

	//すべての石が8近傍で連結していれば、このやり方ですべての石にマークをつけられる。
	return mark == b;
}

bool IsConnected_naive(const uint64_t b) {

	const int dir[9] = { 1,0,-1,-1,0,1,-1,1,1 };

	int searched[100] = { 0 };

	const auto IsStand = [&b](const int x, const int y) {
		if (x <= 0 || 9 <= x || y <= 0 || 9 <= y)return false;
		int num = (y - 1) * 8 + (x - 1);
		return (b & (1ULL << uint64_t(num))) != 0;
	};

	assert(IsStand(4, 4));

	std::stack<std::pair<int, int>>dfs;
	dfs.push(std::make_pair(4, 4));
	while (!dfs.empty()) {
		const auto pos = dfs.top();
		dfs.pop();
		const int x = pos.first;
		const int y = pos.second;
		if (searched[y * 10 + x]++)continue;
		if (!IsStand(x, y))continue;
		for (int i = 0; i < 8; ++i) {
			dfs.push(std::make_pair(x + dir[i], y + dir[i + 1]));
		}
	}

	for (int x = 1; x <= 8; ++x)for (int y = 1; y <= 8; ++y) {
		if (IsStand(x, y) && searched[y * 10 + x] == 0)return false;
	}
	return true;
}

void Test__IsConnected() {

	std::mt19937_64 rnd(12345);

	assert(IsConnected(0xFFFFFFFFFFFFFFFFULL));
	assert(IsConnected_naive(0xFFFFFFFFFFFFFFFFULL));

	for (int ite = 0; ite < 100000; ++ite) {
		std::cout << ite << std::endl;
		uint64_t bb = 0xFFFFFFFFFFFFFFFFULL;
		for(int i = 0; i < 20; ++i) {
			bb &= rnd() | rnd() | rnd();
			bb |= 0x0000'0018'1800'0000ULL;
			auto ans1 = IsConnected(bb);
			auto ans2 = IsConnected_naive(bb);
			if (ans1 != ans2) {
				auto ans11 = IsConnected(bb);
				auto ans22 = IsConnected_naive(bb);
				assert(false);
			}
		}
		bb = 0x0000'0018'1800'0000ULL;
		for (int i = 0; i < 20; ++i) {
			bb |= rnd() & rnd() & rnd();
			auto ans1 = IsConnected(bb);
			auto ans2 = IsConnected_naive(bb);
			if (ans1 != ans2) {
				auto ans11 = IsConnected(bb);
				auto ans22 = IsConnected_naive(bb);
				assert(false);
			}
		}
	}
}

//posはopponentの石の座標を示しているとする。
//現在局面がplayerの手番だとして、それが「直前にopponentがposに石を置いたからこうなった」と仮定したときに、
//その着手によってひっくり返った石のビットボードとしてありうるものを列挙してresultに入れる。個数を返り値とする。
//ただし返り値が非ゼロのときresult[0]=0で、これは便宜上そうしているだけ。返り値はこれを含めているので1引くべし。
int retrospective_flip(uint64_t pos, uint64_t player, uint64_t opponent, std::array<uint64_t, 10000> &result) {

	assert(pos < 64);
	assert((1ULL << pos) & opponent);
	assert(((1ULL << pos) & 0x0000001818000000ULL) == 0);

	int answer = 0;

	const int xpos = pos % 8;
	const int ypos = pos / 8;

	//上方向
	if(ypos >= 2){
		int length = 0;
		while (1) {
			if ((1ULL << (pos - ((length + 1) * 8))) & opponent)++length;
			else break;
			if (length == ypos)break;
		}

		//この時点で、上方向にlength個の連続したopponentの石があると判明している。

		if (length >= 2) {

			//上方向に0～(length-1)個の石をひっくり返したという可能性があるので、それらのビットボードを格納しておく。

			result[0] = 0;
			answer = 1;
			for (int i = 1; i < length; ++i) {
				result[answer] = result[answer - 1] | (1ULL << (pos - (i * 8)));
				++answer;
			}
		}
	}

	//下方向
	if (ypos < 6) {
		int length = 0;
		while (1) {
			if ((1ULL << (pos + ((length + 1) * 8))) & opponent)++length;
			else break;
			if (length == 7 - ypos)break;
		}

		//この時点で、下方向にlength個の連続したopponentの石があると判明している。

		if (length >= 2) {

			//下方向に0～(length-1)個の石をひっくり返したという可能性があるので、それらのビットボードを格納しておく。

			if (answer == 0) {
				result[0] = 0;
				answer = 1;
				for (int i = 1; i < length; ++i) {
					result[answer] = result[answer - 1] | (1ULL << (pos + (i * 8)));
					++answer;
				}
			}
			else {
				const int old_answer = answer;
				uint64_t direction = 0;
				for (int i = 1; i < length; ++i) {
					direction |= (1ULL << (pos + (i * 8)));
					for (int j = 0; j < old_answer; ++j) {
						result[answer++] = result[j] | direction;
					}
				}
			}
		}
	}

	//右方向(最下位ビットが右上で、横に連続しているとして。これはA1～H1などといったオセロの記法とは左右反転している。以下同様)
	if (xpos >= 2) {
		int length = 0;
		while (1) {
			if ((1ULL << (pos - (length + 1))) & opponent)++length;
			else break;
			if (length == xpos)break;
		}

		//この時点で、右方向にlength個の連続したopponentの石があると判明している。

		if (length >= 2) {

			//右方向に0～(length-1)個の石をひっくり返したという可能性があるので、それらのビットボードを格納しておく。

			if (answer == 0) {
				result[0] = 0;
				answer = 1;
				for (int i = 1; i < length; ++i) {
					result[answer] = result[answer - 1] | (1ULL << (pos - i));
					++answer;
				}
			}
			else {
				const int old_answer = answer;
				uint64_t direction = 0;
				for (int i = 1; i < length; ++i) {
					direction |= (1ULL << (pos - i));
					for (int j = 0; j < old_answer; ++j) {
						result[answer++] = result[j] | direction;
					}
				}
			}
		}
	}

	//左方向
	if (xpos < 6) {
		int length = 0;
		while (1) {
			if ((1ULL << (pos + (length + 1))) & opponent)++length;
			else break;
			if (length == 7 - xpos)break;
		}

		//この時点で、左方向にlength個の連続したopponentの石があると判明している。

		if (length >= 2) {

			//左方向に0～(length-1)個の石をひっくり返したという可能性があるので、それらのビットボードを格納しておく。

			if (answer == 0) {
				result[0] = 0;
				answer = 1;
				for (int i = 1; i < length; ++i) {
					result[answer] = result[answer - 1] | (1ULL << (pos + i));
					++answer;
				}
			}
			else {
				const int old_answer = answer;
				uint64_t direction = 0;
				for (int i = 1; i < length; ++i) {
					direction |= (1ULL << (pos + i));
					for (int j = 0; j < old_answer; ++j) {
						result[answer++] = result[j] | direction;
					}
				}
			}
		}
	}

	//右上方向
	if (xpos >= 2 && ypos >= 2) {
		int length = 0;
		while (1) {
			if ((1ULL << (pos - ((length + 1) * 9))) & opponent)++length;
			else break;
			if (length == std::min(xpos, ypos))break;
		}

		//この時点で、右上方向にlength個の連続したopponentの石があると判明している。

		if (length >= 2) {

			//右上方向に0～(length-1)個の石をひっくり返したという可能性があるので、それらのビットボードを格納しておく。

			if (answer == 0) {
				result[0] = 0;
				answer = 1;
				for (int i = 1; i < length; ++i) {
					result[answer] = result[answer - 1] | (1ULL << (pos - (i * 9)));
					++answer;
				}
			}
			else {
				const int old_answer = answer;
				uint64_t direction = 0;
				for (int i = 1; i < length; ++i) {
					direction |= (1ULL << (pos - (i * 9)));
					for (int j = 0; j < old_answer; ++j) {
						result[answer++] = result[j] | direction;
					}
				}
			}
		}
	}

	//左下方向
	if (xpos < 6 && ypos < 6) {
		int length = 0;
		while (1) {
			if ((1ULL << (pos + ((length + 1) * 9))) & opponent)++length;
			else break;
			if (length == std::min(7 - xpos, 7 - ypos))break;
		}

		//この時点で、左下方向にlength個の連続したopponentの石があると判明している。

		if (length >= 2) {

			//左下方向に0～(length-1)個の石をひっくり返したという可能性があるので、それらのビットボードを格納しておく。

			if (answer == 0) {
				result[0] = 0;
				answer = 1;
				for (int i = 1; i < length; ++i) {
					result[answer] = result[answer - 1] | (1ULL << (pos + (i * 9)));
					++answer;
				}
			}
			else {
				const int old_answer = answer;
				uint64_t direction = 0;
				for (int i = 1; i < length; ++i) {
					direction |= (1ULL << (pos + (i * 9)));
					for (int j = 0; j < old_answer; ++j) {
						result[answer++] = result[j] | direction;
					}
				}
			}
		}
	}

	//左上方向
	if (xpos < 6 && ypos >= 2) {
		int length = 0;
		while (1) {
			if ((1ULL << (pos - ((length + 1) * 7))) & opponent)++length;
			else break;
			if (length == std::min(7 - xpos, ypos))break;
		}

		//この時点で、左上方向にlength個の連続したopponentの石があると判明している。

		if (length >= 2) {

			//左上方向に0～(length-1)個の石をひっくり返したという可能性があるので、それらのビットボードを格納しておく。

			if (answer == 0) {
				result[0] = 0;
				answer = 1;
				for (int i = 1; i < length; ++i) {
					result[answer] = result[answer - 1] | (1ULL << (pos - (i * 7)));
					++answer;
				}
			}
			else {
				const int old_answer = answer;
				uint64_t direction = 0;
				for (int i = 1; i < length; ++i) {
					direction |= (1ULL << (pos - (i * 7)));
					for (int j = 0; j < old_answer; ++j) {
						result[answer++] = result[j] | direction;
					}
				}
			}
		}
	}

	//右下方向
	if (xpos >= 2 && ypos < 6) {
		int length = 0;
		while (1) {
			if ((1ULL << (pos + ((length + 1) * 7))) & opponent)++length;
			else break;
			if (length == std::min(xpos, 7 - ypos))break;
		}

		//この時点で、右下方向にlength個の連続したopponentの石があると判明している。

		if (length >= 2) {

			//右下方向に0～(length-1)個の石をひっくり返したという可能性があるので、それらのビットボードを格納しておく。

			if (answer == 0) {
				result[0] = 0;
				answer = 1;
				for (int i = 1; i < length; ++i) {
					result[answer] = result[answer - 1] | (1ULL << (pos + (i * 7)));
					++answer;
				}
			}
			else {
				const int old_answer = answer;
				uint64_t direction = 0;
				for (int i = 1; i < length; ++i) {
					direction |= (1ULL << (pos + (i * 7)));
					for (int j = 0; j < old_answer; ++j) {
						result[answer++] = result[j] | direction;
					}
				}
			}
		}
	}

	return answer;
}

std::set<std::array<uint64_t, 2>>retrospective_searched;

std::vector<std::array<uint64_t, 10000>>retroflips;

bool retrospective_search(const Board &board, const bool from_pass) {

	const auto uni = board.unique();
	const auto num_disc = board.popcount();

	if (num_disc <= DISCS) {
		if (leafnode.find(uni) != leafnode.end()) {
			std::cout << "info: found unique board in leafnodes:" << std::endl;
			std::cout << "unique player = " << uni[0] << std::endl;
			std::cout << "unique opponent = " << uni[1] << std::endl;
			std::cout << "board player = " << board.player << std::endl;
			std::cout << "board opponent = " << board.opponent << std::endl;
			return true;
		}
		return false;
	}

	if (retrospective_searched.find(uni) != retrospective_searched.end())return false;
	retrospective_searched.insert(uni);

	const uint64_t occupied = board.player | board.opponent;
	if (!IsConnected(occupied))return false;

	Board prev;

	if (from_pass == false) {
		if (get_moves(board.opponent, board.player) == 0) {
			prev.player = board.opponent;
			prev.opponent = board.player;
			const bool result = retrospective_search(prev, true);
			if (result) {
				std::cout << "pass" << std::endl;
				return true;
			}
		}
	}

	uint64_t b = board.opponent & (~0x0000001818000000ULL);
	if (b == 0)return false;

	int searched = 0;

	for (unsigned long index = 0; _BitScanForward64(&index, b); b &= b - 1) {

		const int num = retrospective_flip(index, board.player, board.opponent, retroflips[num_disc]);
		if (num)searched += num - 1;

		for (int i = 1; i < num; ++i) {
			const uint64_t flipped = retroflips[num_disc][i];
			assert(flipped);
			prev.player = board.opponent ^ (flipped | (1ULL << index));
			prev.opponent = board.player ^ flipped;
			const bool result = retrospective_search(prev, false);
			if (result) {
				std::cout << index << std::endl;
				return true;
			}
		}
	}

	return false;
}

std::string board_print(const Board &board) {

	int i, j, square, x;
	const char *color = "?*O-." + 1;
	uint64_t moves = get_moves(board.player, board.opponent);
	std::string answer;

	answer += "  A B C D E F G H\n";
	for (i = 0; i < 8; ++i) {
		answer += char(i + '1');
		answer += " ";
		for (j = 0; j < 8; ++j) {
			x = i * 8 + j;
			/*if (player == BLACK)*/ square = 2 - ((board.opponent >> x) & 1) - 2 * ((board.player >> x) & 1);
			/*else square = 2 - ((board.player >> x) & 1) - 2 * ((board.opponent >> x) & 1);*/
			if (square == 0 && (moves & (1ULL << x))) ++square;
			answer += color[square];
			answer += " ";
		}
		answer += char(i + '1');
		const auto twodigits = [](const int x) {return std::string(x < 10 ? "0" : "") + std::to_string(x); };
		if (i == 1) {
			answer += std::string(" ") + char(i + '1') + std::string(" to move");
		}
		else if (i == 3) {
			answer += std::string(" ") +
				color[0] + std::string(": player,   discs = ") +
				twodigits(_mm_popcnt_u64(board.player)) + std::string("    moves = ") +
				twodigits(_mm_popcnt_u64(get_moves(board.player, board.opponent)));
		}
		else if (i == 4) {
			answer += std::string(" ") +
				color[1] + std::string(": opponent, discs = ") +
				twodigits(_mm_popcnt_u64(board.opponent)) + std::string("    moves = ") +
				twodigits(_mm_popcnt_u64(get_moves(board.opponent, board.player)));
		}
		else if (i == 5) {
			answer += std::string("  empties = ") +
				twodigits(64 - _mm_popcnt_u64(board.opponent | board.player)) + std::string("      ply = ") +
				twodigits(_mm_popcnt_u64(board.opponent | board.player) - 3);
		}
		answer += "\n";
	}
	answer += "  A B C D E F G H\n";
	return answer;
}

std::string bitboard_write(const uint64_t b)
{
	std::string answer;
	std::string color = ".X";

	answer += "  A B C D E F G H\n";

	for (int i = 0; i < 8; ++i) {
		answer += char(i + '1');
		answer += " ";
		for (int j = 0; j < 8; ++j) {
			int x = i * 8 + j;
			answer += color[((b >> (unsigned)x) & 1)];
			answer += " ";
		}
		answer += char(i + '1');
		answer += "\n";
	}
	answer += "  A B C D E F G H\n";
	return answer;
}

Board random_game(const int seed, const int discs) {

	assert(discs <= 64);

	std::mt19937_64 rnd(seed);

	while (1) {
		Board board = Board::initial();
		for (int i = 0;; ++i) {

			if (board.popcount() >= discs)return board;

			uint64_t b = get_moves(board.player, board.opponent);
			const auto num = _mm_popcnt_u64(b);
			if (num == 0)break;
			for (int p = std::uniform_int_distribution<int>(0, num - 1)(rnd); p; --p)b &= b - 1;
			unsigned long index = 0;
			_BitScanForward64(&index, b);
			const uint64_t flipped = flip(index, board.player, board.opponent);
			Board next;
			next.player = board.opponent ^ flipped;
			next.opponent = board.player ^ (flipped | (1ULL << index));
			board = next;
		}

	}

	return Board::initial();
}

std::vector<std::array<uint64_t, 2>>move34s() {

	const uint64_t examples[3200] = {
0x0000341c10261000,0x006a4a620a406e00,
0x0008640a38240000,0x00760250465a5200,
0x00202c485c242000,0x005e4016004a4e00,
0x0008640a30140400,0x007602700a622a00,
0x0008640a38240000,0x0076025046523600,
0x00102c481c242000,0x006e4016404a4e00,
0x00202c0444261000,0x005e405a1a406e00,
0x00203c0c50261000,0x005642500e406e00,
0x001026501c240000,0x006e400a625a6a00,
0x0004340832140400,0x007a027208622a00,
0x001026101c340000,0x006e400a624a6a00,
0x001026501c240000,0x006e400a625a5200,
0x0000280c502c2000,0x005656700e405e00,
0x001026501c240000,0x006e400a624a6a00,
0x001026501c240000,0x006e400a625a5a00,
0x0000203c4c261000,0x00545e4012406e00,
0x001026501c240000,0x006e400a625a2a00,
0x0004340a32140400,0x007a027008622a00,
0x000024300a640800,0x0076524e50027600,
0x0008640a382c0000,0x0076025046525600,
0x000434102e080000,0x007a026a10762a00,
0x0008640a20340400,0x007602505a422a00,
0x0000341c10061000,0x006a4a226a206e00,
0x0008640a30140400,0x007602500a622a00,
0x00102458082c2000,0x006c4a0650425e00,
0x000434100a240800,0x007a420a70523600,
0x00202c4838280000,0x005e401644565400,
0x00002c3808640800,0x0056524650027600,
0x0008640a38240000,0x0076025046525600,
0x00002c3808600800,0x0056524456047600,
0x001026500c2c2000,0x006e400a52425400,
0x0010244838280000,0x006e481644565600,
0x001026501c240000,0x006e400a625a4a00,
0x00202c481c242000,0x005e4016404a4e00,
0x0008640a30340400,0x007602504a422a00,
0x0008640a38340000,0x00760250464a2a00,
0x0000341c10320400,0x006a4a226a007a00,
0x0000241c50261000,0x006e4a600a406e00,
0x000434123a240400,0x007a026800527200,
0x0000141c10261000,0x006a6a226a006e00,
0x0000341c12640800,0x006a4a6208027600,
0x0020280c502c1000,0x005446500e406e00,
0x00086608382c0000,0x0076005244525600,
0x0008640a20340400,0x007602505a426a00,
0x00202c0878080000,0x005e405604765400,
0x001846203c240000,0x0066205a02527600,
0x00002c3808661000,0x0056524652006e00,
0x0008640a30340400,0x007602500a423a00,
0x00102648382c0000,0x006e401246525400,
0x000434103e080000,0x007a026a00762a00,
0x0004340a30140400,0x007a02700a622a00,
0x0018620834240000,0x00660456484a6e00,
0x0000240c50261000,0x007652720a406e00,
0x00102c48182c0000,0x006e421066525600,
0x00202c0874100000,0x005e4056086e5600,
0x0008640a20340400,0x007602501a426a00,
0x001026501c240000,0x006e400a625a5400,
0x000864122c2c0000,0x0076024852525600,
0x00202c0450261000,0x005e425a0a406e00,
0x00203c0c50261000,0x005642520a406e00,
0x0020380c50261000,0x005446500a406e00,
0x00202c085c242000,0x005e4056004a4e00,
0x00002c38084c2000,0x0056524650025e00,
0x0008640a38240000,0x0076025046527600,
0x0000243408620800,0x006e4a4856047600,
0x0000341c10261000,0x006a4a226a006e00,
0x0008640a38240000,0x00760250465a4a00,
0x0020284c102c2000,0x005656104e405e00,
0x0008642a30140400,0x007602500a622a00,
0x0008640a38240000,0x00760250464a6a00,
0x001866101c140000,0x0066006a226a2a00,
0x00102458082c2000,0x006e480656405e00,
0x0008640a38240000,0x00760250465a5600,
0x0008640a30240000,0x007602504e527600,
0x000434121a380400,0x007a420860466a00,
0x000434500a2c2000,0x007a420a50425e00,
0x000434101e100000,0x007a026a206e2a00,
0x0008640a20140400,0x007602501a622a00,
0x0008640a30340000,0x007602500e4a6a00,
0x0008640a30240000,0x007602504e4a6a00,
0x000432101c140000,0x007a006a226a2a00,
0x001064083c200000,0x006e0856405e5600,
0x00102c481c242000,0x006e4016404a5c00,
0x00202c485c102000,0x005e4016006e5600,
0x0010264834340000,0x006e40124a4a6a00,
0x0004340836100000,0x007a0272086e2a00,
0x0018660834340000,0x006600524a4a2a00,
0x0008640a38240000,0x00760250464a6c00,
0x00002c3808660800,0x0054524652007600,
0x0008642a20140400,0x007602501a622a00,
0x00202c4848382000,0x005e401614465600,
0x0008640c30261000,0x007602524a406e00,
0x00102640043c2000,0x006e401a5a425a00,
0x0000241c54261000,0x006e4a600a406e00,
0x0008640a20140400,0x007602501a626a00,
0x0020241c482c1000,0x004e4a4016406e00,
0x00102650042c2000,0x006e400a5a425a00,
0x000434120a100400,0x007a0268306e6a00,
0x0008640a30340400,0x007602504a426a00,
0x0008642220140400,0x007602581a622a00,
0x0000341c10661800,0x002a4a624a006600,
0x000024300a640800,0x006c4a4e50027600,
0x00200c50082c2000,0x0056720e50425e00,
0x0008241a10140400,0x007612602a622a00,
0x0008640a30340400,0x007602700a423a00,
0x00202c085c240000,0x005e4056004a6e00,
0x00202c0c54261000,0x005e40520a406e00,
0x00102c4818280000,0x006e401664565600,
0x001026501c2c0000,0x006e400a62525400,
0x0008640a20140400,0x007602701a622a00,
0x001026101c140000,0x006e006a226a2a00,
0x00202c0c50261000,0x005442520a406e00,
0x0008340a30140400,0x007602700a622a00,
0x00204c083c280000,0x005e005640565600,
0x0018660834340000,0x006600524a4a6a00,
0x001024483c280000,0x006e481640565400,
0x00203c0c50261000,0x005442520a406e00,
0x00203c0440261000,0x005a425a1a406e00,
0x0000341c10320400,0x006a4a620a407a00,
0x001866102c2c0000,0x0066004a52525600,
0x0008640a30140400,0x007602500a626a00,
0x000024380a640800,0x006c4a4650027600,
0x0008241a30140400,0x007612600a622a00,
0x00202c106c0c0000,0x005e404e10725600,
0x0008240a30140400,0x007612700a622a00,
0x0008660834340000,0x007600524a4a2a00,
0x0000243c084c2000,0x006e4a4056005e00,
0x0008640a30340400,0x007602500a426a00,
0x0004340832340400,0x007a027208426a00,
0x00202c0850241000,0x005e42500e4a6c00,
0x000434103e200000,0x007a026a005e6a00,
0x00203c0c50261000,0x005a42520a406e00,
0x00002c2c10660800,0x005452524a007600,
0x001026501c240000,0x006e400a624a6c00,
0x0000141c10261000,0x002a6a226a006e00,
0x00202c44502c2000,0x005652180e405e00,
0x00002c3808620800,0x0056524456047600,
0x000024380a640800,0x006e4a4650027600,
0x0008640a30240000,0x007602504e4a6c00,
0x00002c2818640800,0x00565256400a7600,
0x000864323c040000,0x00760248027a6a00,
0x0008241a10340400,0x007652006a426a00,
0x0008640a30140400,0x007602700a626a00,
0x0020280450261000,0x005456580e406e00,
0x00102648382c0000,0x006e401046525600,
0x00202c085c200000,0x005e4056005e5400,
0x001026501c2c0000,0x006e400a62525600,
0x0000241c58241000,0x006e4a6006486e00,
0x000434123a240400,0x007a026800526a00,
0x0020280c502c2000,0x005446500e405e00,
0x001026500c282000,0x006e400a50465400,
0x0000141c30461800,0x006a6a224a206600,
0x0000283c084c2000,0x0054564056005e00,
0x00202c0454261000,0x005e405a0a406e00,
0x00202c0878280000,0x005e405604565400,
0x000024300a640800,0x006a4a4e50027600,
0x000434103e100000,0x007a026a006e2a00,
0x001046301c140000,0x006e204a226a2a00,
0x0008640a20140400,0x007602701a626a00,
0x000434300a640800,0x002a424a50027600,
0x0008640a30340400,0x007602504a423a00,
0x00202c48501c2000,0x005e42100e625600,
0x0010660834140000,0x006e00520a6a2a00,
0x001026501c2c0000,0x006e400a60525600,
0x00002c380a640800,0x0056524650027600,
0x0008640a30340400,0x007602700a426a00,
0x00041c202a640800,0x002a621a50027600,
0x0008660c34240000,0x00760052484a6e00,
0x001026500c2c2000,0x006e400a52425c00,
0x0000241c50261000,0x006c4a620a406e00,
0x000434103a240400,0x007a026a00527200,
0x0000203408620800,0x00565e4856047600,
0x001026501c240000,0x006e400a604a6e00,
0x001026500c240000,0x006e400a72523600,
0x000434123a240400,0x007a026800523a00,
0x000432103c140000,0x007a006a026a2a00,
0x001026500c3c2000,0x006e400a52425a00,
0x0010265004282000,0x006e400a58465400,
0x00202c4c502c2000,0x005c42100e405e00,
0x000432103c240000,0x007a006a02527600,
0x0008640a38240000,0x00760250465a5400,
0x0000283808640800,0x0054564456007600,
0x0008640a20340400,0x007602501a423a00,
0x000864121c340000,0x00760248624a6a00,
0x0018462824140000,0x006620521a6a6a00,
0x00102650042c2000,0x006e400a5a425c00,
0x0000241c50261000,0x007652620a406e00,
0x0004381a12340400,0x006a466008427a00,
0x0000343408661800,0x006a4a4a52006600,
0x0000203c084c2000,0x00545e4056005e00,
0x000434300a640800,0x007a420a50027600,
0x00203c0450261000,0x005a425a0a406e00,
0x0000283808640800,0x0056564456007600,
0x001026501c240000,0x006e400a62523600,
0x0008640a20340400,0x007602505a423a00,
0x000024380a640800,0x00565a4650027600,
0x0000143408161000,0x006a6a0a72206e00,
0x000024380a640800,0x0036524650027600,
0x00002c3808661800,0x0054524652006600,
0x00202c504c182000,0x005e400e10665400,
0x0008640a20340400,0x007602505a027a00,
0x00043c2002640800,0x006a425a58027600,
0x001026101c140000,0x006e006a226a6a00,
0x0008642a30340400,0x007602500a426a00,
0x0020245c082c2000,0x004e4a0056405e00,
0x00041c0812340800,0x002a623268027600,
0x000434101e340000,0x007a420a604a6a00,
0x00002c78082c2000,0x0056520650425e00,
0x00202c0c50261000,0x005e40520a406e00,
0x0008642a20340400,0x007602501a426a00,
0x0008640838280000,0x0076005644565400,
0x0008241a10340400,0x007612602a426a00,
0x0000205c082c2000,0x00545e0056405e00,
0x00102650042c2000,0x006e400a5a425400,
0x00002c3808640800,0x0056524456007600,
0x00204c0838280000,0x005e005644565400,
0x0000243c086c0800,0x00565a4056007600,
0x00002c2c10660800,0x005652524a007600,
0x00202c485c242000,0x005e4016004a5600,
0x0004340836140000,0x007a0272086a2a00,
0x0000241c50261000,0x005652620a406e00,
0x00202c4c102c2000,0x005642104e405e00,
0x001866083c240000,0x00660056404a6e00,
0x001026500c2c2000,0x006e400a50425600,
0x0000341c10660800,0x006a4a224a007600,
0x00204c08382c0000,0x005e025046525600,
0x0000241c50261000,0x006a4a620a406e00,
0x00202c0c50261000,0x005652500a406e00,
0x0008341238240400,0x0076026802527200,
0x0000341c10260800,0x006a4a620a507600,
0x0008640a34340000,0x007602500a4a6a00,
0x001026500c2c2000,0x006e400a52425600,
0x000434101a240800,0x007a026a60127600,
0x0004341238240400,0x007a026802527200,
0x0020280858241000,0x0054465406486e00,
0x0008640a34140000,0x007602500a6a2a00,
0x00202c106c280000,0x005e404e10565400,
0x00002c3808660800,0x0056524652007600,
0x0008640a38240000,0x00760250464a6e00,
0x00202c087c100000,0x005e4056006e5400,
0x00102450082c2000,0x006e480e50425e00,
0x000414200a640800,0x002a6a1a50027600,
0x00201844482c2000,0x0054661816405e00,
0x00203c0440261000,0x005642581a406e00,
0x0008241a10340400,0x007652006a423a00,
0x0000243c0c620800,0x006e4a4052047600,
0x0000243c4c261000,0x006e4a4012406e00,
0x000432103c040000,0x007a006a027a2a00,
0x000024380a640800,0x006a4a4650027600,
0x0008640a303c0400,0x007602504a422a00,
0x001064102c280000,0x006e084e50565600,
0x001026501c240000,0x006e400a62525600,
0x0020280c50261000,0x005446500a406e00,
0x00002c2c10661800,0x005652524a006600,
0x0008640a20140400,0x007602501a6a2a00,
0x0000341c10661000,0x006a4a620a006e00,
0x00102458082c2000,0x006e4a0056425c00,
0x00204c083c200000,0x005e0056405e5400,
0x00002c1c482c2000,0x0056526016405e00,
0x0020380c50261000,0x005646500a406e00,
0x0004360834140000,0x007a00720a6a2a00,
0x00102650042c2000,0x006e400a5a425e00,
0x00200c58082c2000,0x0056720650425e00,
0x0008642a30140400,0x007602500a626a00,
0x001026300c640800,0x006e404a52027600,
0x00202c0c50261000,0x005e42520a406e00,
0x00102650042c2000,0x006e400a58425600,
0x00002c2c10661800,0x005452524a006600,
0x0008640a30240000,0x007602504e4a6e00,
0x0000241c50261000,0x006e4a620a406e00,
0x000862083c280000,0x0076045640565400,
0x00203c0450261000,0x0056425a0a406e00,
0x001026501c340000,0x006e400a624a2a00,
0x000434103e080000,0x007a026a00766a00,
0x000434103a040000,0x007a026a007a2a00,
0x001026501c240000,0x006e400a624a6e00,
0x0008640a38240000,0x00760250465a5a00,
0x00204c083c280000,0x005e005640565400,
0x0000243408621800,0x006e4a4856046600,
0x0008640a382c0000,0x0076025046525400,
0x0004340836100000,0x007a0272086e6a00,
0x00086008382c0000,0x0076045644525600,
0x0008642a20140400,0x007602501a626a00,
0x000824121c140000,0x00761268226a2a00,
0x0008640a20340400,0x007602505a425a00,
0x00202c087c200000,0x005e4056005e5400,
0x000024300a640800,0x006e4a4e50027600,
0x000434103e280000,0x007a026a00566a00,
0x001046303c040000,0x006e204a027a2a00,
0x00043c300a640800,0x002a424a50027600,
0x0000243c086c0800,0x006e4a4056007600,
0x0004341238240000,0x007a026806527600,
0x0008640a38140000,0x00760250066a2a00,
0x00002c3808660800,0x0056524650007600,
0x00102c4c102c2000,0x006c42104e405e00,
0x0000243c086c2000,0x006e4a4056005e00,
0x0008620c38280000,0x0076045244565400,
0x00201c58482c2000,0x0056620610425e00,
0x000434202a640800,0x007a025a50027600,
0x0000143408660800,0x006a6a0a52007600,
0x000434300a340800,0x006a4a0a70027600,
0x0010660834340000,0x006e00524a4a2a00,
0x000434121a340400,0x007a4208604a6a00,
0x00002838084c2000,0x0054564456005e00,
0x0000243c084c2000,0x00565a4056005e00,
0x00202c0858241000,0x005e4250064a6c00,
0x001016301c140000,0x006e204a226a2a00,
0x0008640a20340400,0x007602701a426a00,
0x00002c3c084c2000,0x0056524056005e00,
0x00202c0874100000,0x005e4056086e5400,
0x00002838084c2000,0x0056564456005e00,
0x001006101c340000,0x006e206a224a6a00,
0x001064083c200000,0x006e0856405e5400,
0x000864121c340000,0x00760248624a2a00,
0x0000203c084c2000,0x00565e4056005e00,
0x0008642a20340400,0x007602505a027a00,
0x00086408382c0000,0x0076005644525600,
0x0000245c102c2000,0x006e4a204e405e00,
0x0000343408660800,0x006a4a4a52007600,
0x001026501c340000,0x006e400a624a6a00,
0x001026502c2c0000,0x006e400a52525400,
0x00002c38084c2000,0x0056524456005e00,
0x001026501c280000,0x006e400a60565600,
0x0010264834340000,0x006e40124a4a2a00,
0x000434300a640800,0x007a424a50027600,
0x0000043c10320400,0x002a7a026a007a00,
0x0000241c50261000,0x003652620a406e00,
0x0008640a34340000,0x007602504a4a6a00,
0x00041c300a340800,0x006a620a70027600,
0x00202c0858241000,0x005e4056044a6c00,
0x00102c500c2c2000,0x006e400e50425c00,
0x0008640a20340400,0x007602701a423a00,
0x0008640a34340000,0x007602504a4a2a00,
0x001026500c2c2000,0x006e400a52425a00,
0x00043c200a640800,0x006a425a50027600,
0x0000240c50261000,0x006e4a720a406e00,
0x000434300a640800,0x006a4a4a50027600,
0x00202c087c080000,0x005e405600765400,
0x00202c504c282000,0x005e400e10465400,
0x00202c087c200000,0x005e4056005e5600,
0x000434101a240800,0x003a426a00527600,
0x000434101a240800,0x007a420a60523600,
0x001026500c240000,0x006e400a72525600,
0x00102458082c2000,0x006e4a0056405e00,
0x000434102e080000,0x007a026a10766a00,
0x00204c083c240000,0x005e0056404a6e00,
0x0008241a10140400,0x007612602a626a00,
0x0004341238140400,0x007a0268026a2a00,
0x00202c0450261000,0x0054425a0a406e00,
0x000414202a640800,0x002a621a50027600,
0x001026500c240000,0x006e400a724a6e00,
0x0000243c44261000,0x006e4a401a406e00,
0x00202c485c242000,0x005e4016004a5c00,
0x0008241a10340400,0x007612600a427a00,
0x0008642220140400,0x007602581a626a00,
0x000436103c240000,0x007a006a02527600,
0x00202c0858241000,0x005e425006486e00,
0x0000341418260800,0x006a4a6a02507600,
0x000434200a640800,0x002a425a50027600,
0x0000141812340800,0x002a6a2668027600,
0x00202c087c240000,0x005e4056004a6e00,
0x00202c104c2c1000,0x005e404e10426c00,
0x0020280c50241000,0x005446500e486e00,
0x0000283c084c2000,0x0056564056005e00,
0x000434103a040000,0x007a026a047a6a00,
0x0020280c50261000,0x005456500a406e00,
0x00202c087c040000,0x005e4056007a5600,
0x0000243c10260800,0x007652026a107600,
0x0008642220340400,0x007602581a426a00,
0x0004340a32140400,0x007a027008626a00,
0x0000043c10320400,0x006a7a026a007a00,
0x0000241c502c2000,0x006e4a600e405e00,
0x0004340a32340400,0x007a027008423a00,
0x0008640a30340400,0x007602504a425a00,
0x00202c085c241000,0x005e4056004a6c00,
0x000414200a640800,0x002a621a70027600,
0x000866101c340000,0x0076000a624a6a00,
0x00002c2c50261000,0x005652500a406e00,
0x00203c4c502c2000,0x005642100e405e00,
0x0018660834140000,0x006600520a6a2a00,
0x0008640a30140000,0x007602500e6a2a00,
0x00202c0450261000,0x0054525a0a406e00,
0x001026101c340000,0x006e006a224a6a00,
0x00043c200a640800,0x002a425a50027600,
0x0020245c482c2000,0x004e4a0016405e00,
0x0008640a30340400,0x007602504a027a00,
0x000434101e140000,0x007a026a206a2a00,
0x000434103e040000,0x007a026a007a6a00,
0x0000343408661800,0x002a4a4a52006600,
0x0004340a30340000,0x007a02700e4a6a00,
0x000864321c140000,0x00760248226a2a00,
0x0000243c10320400,0x007652026a007a00,
0x0008640838280000,0x0076005644565600,
0x0004341238240400,0x007a026802526a00,
0x0008241a30140400,0x007612600a626a00,
0x0000247c082c2000,0x006e4a0056405e00,
0x0000240c50261000,0x005652720a406e00,
0x0004340a32340400,0x007a027008426a00,
0x00202c4c102c2000,0x005c42104e405e00,
0x0000243c0c621800,0x006e4a4052046600,
0x0000143c10320400,0x002a6a026a007a00,
0x00202c085c240000,0x005e4056005a5600,
0x00002c38482c2000,0x0056524610425e00,
0x0000240c50261000,0x003652720a406e00,
0x0000141c10320400,0x006a6a226a007a00,
0x00202c0450261000,0x005c42580a406e00,
0x001026541c240000,0x006e400a604a6e00,
0x00202c087c140000,0x005e4056006a5600,
0x000432101c140000,0x007a006a226a6a00,
0x00002c3808661800,0x0056524452006600,
0x000864121c340000,0x00760248224a6a00,
0x00204c083c200000,0x005e0056405e5600,
0x0000282c10620800,0x005456504e047600,
0x000864122c2c0000,0x0076024852525400,
0x00203c0440261000,0x0056425a1a406e00,
0x0008341238040000,0x00760268067a6a00,
0x000434302a640800,0x007a024a50027600,
0x00102458182c0000,0x006e4a0066525600,
0x0000243c0c661800,0x006e4a4052006600,
0x001066102c2c0000,0x006e004a52525400,
0x0000343408660800,0x002a4a4a52007600,
0x0008341238240400,0x0076026802526a00,
0x001026500c2c2000,0x006e400a50425c00,
0x0008340a30340400,0x007602700a426a00,
0x001026500c2c0000,0x006e400a72525400,
0x00002c2c10661000,0x005452524a006e00,
0x0008341238040000,0x00760268067a2a00,
0x00202c085c200000,0x005e4056005e5600,
0x0020280c54261000,0x005446500a406e00,
0x000434103a240400,0x007a026a00523a00,
0x00202c504c282000,0x005e400e10465600,
0x0008640a20140400,0x007602701a6a2a00,
0x0018462834140000,0x006620520a6a2a00,
0x00002c380a640800,0x0054524650027600,
0x00186608382c0000,0x0066005246525400,
0x0020280458241000,0x0056565806486e00,
0x001026500c282000,0x006e400a50465600,
0x0008642a30340400,0x007602500a423a00,
0x000434102e280000,0x007a026a10566a00,
0x0000245c082c2000,0x006e4a0056405e00,
0x000414320a340400,0x002a620870027a00,
0x000024380a640800,0x0056524650027600,
0x00203c0450261000,0x0054425a0a406e00,
0x0010264c3c240000,0x006e4012404a6e00,
0x00002c3808661800,0x0056524652006600,
0x00202c105c240000,0x005e404e204a6e00,
0x00202c0c50261000,0x005a42520a406e00,
0x000434103e100000,0x007a026a006e6a00,
0x00102450082c2000,0x006c4a0e50425e00,
0x0000245c082c2000,0x006e4a2056405e00,
0x0000341c10660800,0x006a4a624a007600,
0x000024340c661800,0x006e4a4852006600,
0x00102458082c2000,0x006e4a0056425600,
0x0000341812340400,0x006a4a6608427a00,
0x000434101a240800,0x007a026a00527600,
0x00106608382c0000,0x006e005246525400,
0x00202c0450261000,0x005e405a0a406e00,
0x000434103e140000,0x007a026a006a2a00,
0x00202c0c502c1000,0x005652500e406e00,
0x0020280c50261000,0x005656500a406e00,
0x000432103c040000,0x007a006a027a6a00,
0x0008640a30340000,0x007602504e4a2a00,
0x0010265004282000,0x006e400a58465600,
0x00202c104c282000,0x005e404e10465400,
0x00002c2c12640800,0x0056525248027600,
0x000868083c200000,0x00760456405e5600,
0x0000341c10661800,0x006a4a624a006600,
0x0000343c10320400,0x006a4a026a007a00,
0x0008640a38040000,0x00760250067a2a00,
0x00102650042c2000,0x006e400a58425c00,
0x000414300a240800,0x006a6a0a70127600,
0x001026500c382000,0x006e400a50465400,
0x000434103a240400,0x007a026a00526a00,
0x0000243c10320400,0x006a5a026a007a00,
0x000434103a240800,0x007a026a00523600,
0x00041c300a240800,0x002a620a70127600,
0x00204c0838280000,0x005e005644565600,
0x0008243a10340400,0x003652006a027a00,
0x0000143c10320400,0x006a6a026a007a00,
0x001862043c240000,0x0066045a404a6e00,
0x0008340a30140400,0x007602700a626a00,
0x000434103a040000,0x007a026a007a6a00,
0x00043c2002640800,0x002a425a58027600,
0x0004341218140000,0x007a0268266a2a00,
0x0010264c34240000,0x006e4012484a6e00,
0x0000341c10660800,0x002a4a624a007600,
0x00000c78082c2000,0x0056720650425e00,
0x0004340836300000,0x007a0272084e6a00,
0x00203c0440261000,0x0054425a1a406e00,
0x00086408382c0000,0x0076025046525600,
0x001026500c240000,0x006e400a72527600,
0x00202c481c240000,0x005e4016604a6e00,
0x0004340832140400,0x007a027208626a00,
0x0010245808282000,0x006e480654465400,
0x001866043c240000,0x0066005a404a6e00,
0x0008660834140000,0x007600520a6a2a00,
0x00002c3808661800,0x0056524650006600,
0x000434103a200000,0x007a026a045e6a00,
0x001026502c280000,0x006e400a50565400,
0x0004340a30140400,0x007a02700a626a00,
0x0008642a20340400,0x007602501a423a00,
0x001026500c282000,0x006e400a50565400,
0x0008640a20340400,0x007602505a427a00,
0x00202c085c242000,0x005e4056004a5c00,
0x0000343408660800,0x006a4a0a52007600,
0x0008241a30340400,0x007612600a426a00,
0x00202c085c040000,0x005e4056007a5600,
0x001026500c282000,0x006e400e50465400,
0x00202c0878080000,0x005e405604765600,
0x000866101c340000,0x0076004a624a2a00,
0x00202c48481c2000,0x005e421016625600,
0x00202c106c080000,0x005e404e10765400,
0x001066101c340000,0x006e004a224a6a00,
0x00202c0c50261000,0x005652520a406e00,
0x000866101c140000,0x0076006a226a2a00,
0x0008640a203c0400,0x007602505a422a00,
0x00202c0450261000,0x0056525a0a406e00,
0x0008241a14140000,0x007612602a6a2a00,
0x0000243c0c660800,0x006e4a4052007600,
0x0000141c10320400,0x002a6a226a007a00,
0x00002c0858241000,0x00565276004a6e00,
0x001016101c340000,0x006e206a224a6a00,
0x0018660834340000,0x006600520a4a6a00,
0x001846301c140000,0x0066204a226a2a00,
0x00202c085c240000,0x005e4056204a6e00,
0x001046300c140000,0x006e204a326a2a00,
0x000432101c340000,0x007a006a224a6a00,
0x0008640a30340400,0x007602504a4a2a00,
0x00102654042c2000,0x006e400a5a405e00,
0x001026540c2c2000,0x006e400a50425600,
0x0020280858241000,0x0054565406486e00,
0x0004243a10340400,0x003a52006a027a00,
0x0000282c10661800,0x005456504a006600,
0x0020241c482c2000,0x004e4a4016405e00,
0x000434101a240800,0x007a026a20523600,
0x00043c101a240800,0x006a426a00527600,
0x0000243c04620800,0x006e4a405a047600,
0x0000203408621800,0x00545e4856046600,
0x0020280c58241000,0x0054465006486e00,
0x00102458082c2000,0x006e480650425e00,
0x001026501c240000,0x006e400a625a5600,
0x00102644042c2000,0x006e401a5a405e00,
0x0008241a18140000,0x00761260266a2a00,
0x00202c500c282000,0x005e400e50465400,
0x000432101c340000,0x007a400a624a6a00,
0x00000c70082c2000,0x0056720e50425e00,
0x0020241c482c2000,0x00564a4016405e00,
0x0008241a10340400,0x003652206a027a00,
0x00202c08780c0000,0x005e425006725600,
0x000434121c140000,0x007a0268226a2a00,
0x0008241a10340400,0x003652600a427a00,
0x00202c0450261000,0x005c425a0a406e00,
0x0000243c10360800,0x007652026a007600,
0x000434103e240000,0x007a026a00527600,
0x001866301c140000,0x0066004a226a2a00,
0x0000341c10660800,0x006a4a620a007600,
0x001866102c2c0000,0x0066004a52525400,
0x00202c481c302000,0x005e4016404e5400,
0x00041c0812340400,0x006a623268027a00,
0x0000086c102c2000,0x005476104e405e00,
0x0000243c08660800,0x00565a4056007600,
0x0008240a30140400,0x007612700a626a00,
0x00202c0450261000,0x005a425a0a406e00,
0x00202c0858241000,0x005e405604486e00,
0x0020284c102c2000,0x005446104e405e00,
0x00002c3808661000,0x0054524652006e00,
0x0010660834140000,0x006e00520a6a6a00,
0x00202c0c50261000,0x005452520a406e00,
0x0020245c482c2000,0x005c4a0016405e00,
0x00002c2c10661800,0x005652504a006600,
0x001026540c2c2000,0x006e400a52405e00,
0x00002c2c12640800,0x0054525248027600,
0x0004340832340800,0x007a027208423600,
0x0000243c48241000,0x006e4a4016486e00,
0x00202c0c50261000,0x005c42520a406e00,
0x0020284c502c2000,0x005446100e405e00,
0x000024380a640800,0x0076524650027600,
0x0008640a30140000,0x007602500e6a6a00,
0x00043c200a640800,0x006a421a50027600,
0x000866100c140000,0x0076006a326a2a00,
0x001066101c340000,0x006e004a624a2a00,
0x00202c481c200000,0x005e4016605e5400,
0x0004340832340400,0x007a027208423a00,
0x0000283c04621800,0x005656405a046600,
0x00102648382c0000,0x006e401246525600,
0x0008240a30340400,0x007612700a426a00,
0x00002c3808680800,0x0056524456047600,
0x0000043a10340400,0x006a7a046a027a00,
0x0004340a32100400,0x007a0270086e2a00,
0x00102458082c2000,0x006c4a0456405e00,
0x00102c500c282000,0x006e400e50465400,
0x0000243c08621800,0x006e4a4056046600,
0x0004243a10340400,0x007252006a027a00,
0x000434103a240000,0x007a026a045a6a00,
0x001866101c340000,0x0066004a624a6a00,
0x000434300a640800,0x006a424a50027600,
0x0000141418260800,0x002a6a2a62107600,
0x0020280c50241000,0x005456500e486e00,
0x00203c18482c1000,0x0056424610426e00,
0x0000282c50261000,0x005656500a406e00,
0x00202c504c2c2000,0x005e400e10425600,
0x00002c380c680800,0x0056524452047600,
0x001026502c2c0000,0x006e400a52525600,
0x0000243c08620800,0x006e4a4056047600,
0x001026540c2c2000,0x006e400a50425c00,
0x001026501c240000,0x006e400a62527600,
0x0000243c104c2000,0x006e4a404e205e00,
0x0000341c12240800,0x006a4a6208527600,
0x00043c300a640800,0x006a424a50027600,
0x00202c0c50261000,0x005642520a406e00,
0x0008662834140000,0x007600520a6a2a00,
0x00043c300a640800,0x005a424a50027600,
0x0008640a38240000,0x00760250465a2a00,
0x00102c481c200000,0x006e4016605e5600,
0x0004340a30340400,0x007a02700a426a00,
0x0008680c30280000,0x007604524c565400,
0x00002c38482c1000,0x0056524610426e00,
0x00002c2c10661000,0x005652524a006e00,
0x000024380a640800,0x00525a4650027600,
0x000434120a140400,0x007a0268306a2a00,
0x0000243c08660800,0x006e4a4056007600,
0x00040c3812340800,0x006a720268027600,
0x001026500c2c2000,0x006e400a52405e00,
0x00002c3848261000,0x0054524612406e00,
0x001064083c240000,0x006e0856404a6e00,
0x0004340a30340400,0x007a02700a423a00,
0x000434103a240000,0x007a026a00527600,
0x00102458082c0000,0x006e4a0076525600,
0x000866102c2c0000,0x0076004a52525400,
0x000866102c140000,0x0076006a126a2a00,
0x00186608382c0000,0x0066005046525600,
0x001026501c280000,0x006e400a60565400,
0x00201c50482c2000,0x0056620e10425e00,
0x0008642220340400,0x007602581a423a00,
0x0008341238240400,0x0076026802523a00,
0x001024580c282000,0x006e480650465400,
0x00202c58082c2000,0x0056520650425e00,
0x000434121a300400,0x007a4208604e6a00,
0x0020301c482c1000,0x00544e4016406e00,
0x0000282c10661800,0x005656504a006600,
0x001866101c340000,0x0066004a624a2a00,
0x0000243a08340400,0x0076520472027a00,
0x000434103a240000,0x007a026a04527600,
0x0000243c086c1000,0x006e4a4056006e00,
0x0000243c04621800,0x006e4a405a046600,
0x00201844502c2000,0x005466180e405e00,
0x001026483c280000,0x006e401640565600,
0x0000243408680800,0x006e4a4856047600,
0x00106c083c280000,0x006e005640565600,
0x00202c0a50340400,0x005e42500a427a00,
0x00086608382c0000,0x0076005246525400,
0x000434200a640800,0x006a425a50027600,
0x0020280c50261000,0x005646500a406e00,
0x0018660c3c240000,0x00660052404a6e00,
0x0000243408660800,0x006e4a4856007600,
0x0008241a10340400,0x007612602a423a00,
0x000434083a240000,0x007a027204527600,
0x0000243a10340400,0x007652046a027a00,
0x000024340c660800,0x006e4a4852007600,
0x0020280450261000,0x005656580a406e00,
0x0000243c08661800,0x006e4a4056006600,
0x0000341c12640800,0x006a4a6248027600,
0x00202c08780c0000,0x005e405604725600,
0x001026540c282000,0x006e400a50465400,
0x001066301c140000,0x006e004a226a2a00,
0x000034340a640800,0x006a4a4a50027600,
0x0008341238240000,0x0076026806527600,
0x0000241c50261000,0x00525a620a406e00,
0x0008640a30340400,0x007602504a427a00,
0x001066303c140000,0x006e004a026a2a00,
0x000434300a640800,0x003a424a50027600,
0x000434200a640800,0x006a4a5a50027600,
0x00202c0858241000,0x005c4256004a6e00,
0x00002c3808640800,0x00565246500a7600,
0x001026540c282000,0x006e400a50465600,
0x00043c1a12340400,0x006a426008427a00,
0x0000341c18320400,0x006a4a2262047a00,
0x00202c0450261000,0x0056425a0a406e00,
0x0004143208340400,0x002a620872027a00,
0x001866102c280000,0x0066004a50565400,
0x0004341238240400,0x007a026802523a00,
0x00102644042c2000,0x006e401a58425c00,
0x0000043a10340400,0x002a7a006a027a00,
0x0008241a30340400,0x007612600a423a00,
0x0008620c3c240000,0x00760452404a6e00,
0x0004342022640800,0x007a025a58027600,
0x0000243c04660800,0x006e4a405a007600,
0x0000043c10461800,0x006a7a026a206600,
0x0018620c38280000,0x0066045244565400,
0x00186608382c0000,0x0066005244525600,
0x0000241c50261000,0x00545a620a406e00,
0x000862083c240000,0x00760456404a6e00,
0x0000241c50261000,0x004a5a620a406e00,
0x000028380c621800,0x0054564452046600,
0x000414300a340400,0x002a620a70027a00,
0x000866301c140000,0x0076004a226a2a00,
0x00102458082c2000,0x006e480654425c00,
0x00086c102c280000,0x0076004e50565400,
0x00202c0858241000,0x00564256004a6e00,
0x001024581c240000,0x006e4806604a6e00,
0x00002c3808661000,0x0056524650006e00,
0x000434300a640800,0x006a4a0a50027600,
0x000862043c240000,0x0076045a404a6e00,
0x00200854082c2000,0x0054760856405e00,
0x0000243c20461000,0x007652025a206e00,
0x00002c0c50261000,0x005652720a406e00,
0x00002c38086c0800,0x0056524650027600,
0x0008640a303c0400,0x007602504a426a00,
0x000866103c240000,0x0076006a02527600,
0x0008640a301c0400,0x007602500a622a00,
0x001024500c282000,0x006e480e50465400,
0x0020280c50261000,0x005456500e406e00,
0x00200c58482c2000,0x0056720416405e00,
0x000434121a3c0400,0x007a420860426a00,
0x0008620c3c280000,0x0076045240565400,
0x00043c200a640800,0x005a425a50027600,
0x0020280450261000,0x005456580a406e00,
0x00201c58082c2000,0x0056620650425e00,
0x00202c0858241000,0x005e405606486e00,
0x00102650042c2000,0x006e400a5a425600,
0x0020341c482c2000,0x00564a4016405e00,
0x001026500c2c0000,0x006e400a70525600,
0x00201848482c2000,0x0056661416405e00,
0x0018620c34240000,0x00660452484a6e00,
0x000866102c280000,0x0076004a50565400,
0x0008642a30340000,0x007602500e4a6a00,
0x000414200a640800,0x002a621a50027600,
0x00002c3848261000,0x0056524612406e00,
0x00200c58482c2000,0x0056720610425e00,
0x000864320c140000,0x00760248326a2a00,
0x0000243408661800,0x006e4a4856006600,
0x00202c501c240000,0x005e400e604a6e00,
0x00102654042c2000,0x006e400a58425c00,
0x0000243c184c2000,0x006e4a4046205e00,
0x00041c300a640800,0x002a620a50027600,
0x000024340c620800,0x006e4a4852047600,
0x0018620c30280000,0x006604524c565400,
0x001026500c2c0000,0x006e400a72525600,
0x00106c083c240000,0x006e0056404a6e00,
0x0010660834200000,0x006e0056485e5400,
0x0000203c08620800,0x00565e4056047600,
0x00086c083c240000,0x00760056404a6e00,
0x001026500c282000,0x006e400e50465600,
0x000434200a640800,0x003a425a50027600,
0x001026443c240000,0x006e401a404a6e00,
0x00102458082c2000,0x006e480654405e00,
0x000434300a640800,0x007a024a50027600,
0x0000103608340400,0x002a6e0872027a00,
0x0000143408660800,0x002a6a0a52007600,
0x0000341c10661000,0x002a4a624a006e00,
0x00202c0858241000,0x005e4056004a6e00,
0x001016300c140000,0x006e204a326a2a00,
0x0008340a30340400,0x007602700a423a00,
0x0000343448261000,0x006a4a4a12406e00,
0x00202c085c241000,0x005e4056004a6e00,
0x0010265404282000,0x006e400a58465400,
0x000024340c621800,0x006e4a4852046600,
0x0000143c10461800,0x006a6a026a206600,
0x0008640a38340000,0x00760250464a6a00,
0x00102458082c2000,0x006e4a0650425e00,
0x00102c481c240000,0x006e4016604a6e00,
0x000434103a140400,0x007a026a006a2a00,
0x000866300c140000,0x0076004a326a2a00,
0x00043c2002640800,0x005a425a58027600,
0x000434300a640800,0x005a424a50027600,
0x000432083c240000,0x007a047202527600,
0x0008241a10140000,0x007612602e6a2a00,
0x000028300c661800,0x0054564c52006600,
0x0008642a30140000,0x007602500e6a6a00,
0x001026501c240000,0x006e400e604a6e00,
0x000034340a640800,0x006a4a0a50027600,
0x0000243c10461800,0x007652026a206600,
0x0000083e10340400,0x002a76006a027a00,
0x00002c3848261000,0x0056524610406e00,
0x000434200a640800,0x005a425a50027600,
0x000034181a240800,0x006a4a6600527600,
0x00202c104c282000,0x005e404e10465600,
0x0008660834340000,0x007600524a4a6a00,
0x0010264404282000,0x006e401a58465400,
0x0000241c50261000,0x006a5a620a406e00,
0x0020380450261000,0x005646580a406e00,
0x00202c0858241000,0x005e4256004a6e00,
0x00102654042c2000,0x006e400a58425600,
0x00102650043c2000,0x006e400a5a425a00,
0x00102c500c382000,0x006e400e50465600,
0x00102450082c2000,0x006e4a0e50425e00,
0x0000243c20660800,0x007652025a007600,
0x00202c0850241000,0x005e4256084a6e00,
0x0008240a30340400,0x007612700a423a00,
0x00200c5c482c2000,0x0056720016405e00,
0x000434200a640800,0x007a025a50027600,
0x000434200a640800,0x007a425a50027600,
0x00202c0450261000,0x005652580a406e00,
0x0004340a221c0400,0x007a027018622a00,
0x000434123a300400,0x007a0268004e6a00,
0x00202c087c280000,0x005e405600565400,
0x0000241c50261000,0x006e4a600e406e00,
0x00002c1c50261000,0x005452620a406e00,
0x0000243c08680800,0x006e4a4056047600,
0x0000341c10661000,0x006a4a624a006e00,
0x00002c0c50261000,0x005452720a406e00,
0x0004341218340000,0x007a4208664a6a00,
0x0000283e10340400,0x006a56006a027a00,
0x0008640a20340400,0x007602505a4a6a00,
0x0010265004282000,0x006e400e58465400,
0x0004340a32180400,0x007a027008662a00,
0x0008642220340400,0x007602585a027a00,
0x00203c10482c1000,0x0056424e10426e00,
0x00002c38086c2000,0x0056524456005e00,
0x00201c50082c2000,0x0056620e50425e00,
0x000434123c240000,0x007a026802527600,
0x0000282c10660800,0x005456504a007600,
0x0000301e10340400,0x006a4e600a427a00,
0x001026500c2c2000,0x006e400a50425e00,
0x0000147c082c2000,0x00566a0056405e00,
0x0004180a12340400,0x006a663068027a00,
0x00002c18482c1000,0x0056526610426e00,
0x00200c50482c2000,0x0056720e10425e00,
0x0000282c10661000,0x005456504a006e00,
0x0000243404621800,0x00565a485a046600,
0x00102650042c2000,0x006e400a5a405e00,
0x000866303c040000,0x0076004a027a2a00,
0x0000243c482c1000,0x006e4a4016406e00,
0x0000341c10360800,0x006a4a620a407600,
0x0010245808282000,0x006e480654465600,
0x0000341c50261000,0x006a4a620a406e00,
0x0018660c34240000,0x00660052484a6e00,
0x0004243a10340400,0x006a52006a027a00,
0x000434103e040000,0x007a026a007a2a00,
0x0008620c30280000,0x007604524c565400,
0x001866203c240000,0x0066005a02527600,
0x000434300a640800,0x002a4a4a50027600,
0x000434101e100000,0x007a026a206e6a00,
0x0020380450261000,0x005446580a406e00,
0x0000203a10340400,0x006a5e046a027a00,
0x0004340a30140400,0x007a02700a6a6a00,
0x000418320a340400,0x006a660870027a00,
0x0020380440261000,0x005446581a406e00,
0x000434200a640800,0x002a4a5a50027600,
0x000868083c240000,0x00760456404a6e00,
0x0000082e10340400,0x006a76106a027a00,
0x001066102c280000,0x006e004a50565400,
0x0000300e10340400,0x006a4e700a427a00,
0x001026500c3c2000,0x006e400a52425400,
0x0000341c50261000,0x002a4a620a406e00,
0x0008240a10340400,0x003652700a427a00,
0x000434083a040000,0x007a0272047a2a00,
0x000024380a640800,0x00545a4650027600,
0x00002c2818641000,0x00565256400a6e00,
0x00202c504c2c2000,0x005e400e10425c00,
0x0000243c482c2000,0x006e4a4016405e00,
0x000024380a640800,0x006a5a4650027600,
0x00002c2c50261000,0x005452520a406e00,
0x00002c2858241000,0x00565256004a6e00,
0x0000343408661000,0x006a4a4a52006e00,
0x00002c28184c2000,0x0056525640225e00,
0x000434120a180400,0x007a026830662a00,
0x001026500c240000,0x006e400a725a5600,
0x000864121c340000,0x00760208624a6a00,
0x0020280450261000,0x005446580a406e00,
0x0008640a30140400,0x007602500a6a2a00,
0x0000243c18320400,0x0076520262047a00,
0x0000341c12640800,0x002a4a6248027600,
0x00202c104c2c2000,0x005e404e10425c00,
0x00002c3408620800,0x0056524856047600,
0x00202c0450261000,0x005e42580a406e00,
0x000024344c261000,0x006e4a4812406e00,
0x00203c58482c2000,0x0056420610425e00,
0x0020380c50241000,0x005446500e486e00,
0x00201850482c2000,0x0054660c16405e00,
0x000434101a240800,0x006a426a00527600,
0x00102c481c282000,0x006e401640565600,
0x0004340836340000,0x007a0272084a6a00,
0x00202c500c2c0000,0x005e400e70525600,
0x0000143408661800,0x002a6a0a52006600,
0x0004083a12340400,0x002a760068027a00,
0x000836101c340000,0x0076400a624a6a00,
0x001026501c200000,0x006e400a605e5400,
0x000434122a080400,0x007a026810766a00,
0x00204c08382c0000,0x005e005644525600,
0x00002c1c50261000,0x005652620a406e00,
0x000028380c661800,0x0054564452006600,
0x000866101c140000,0x0076006a226a6a00,
0x001024580c2c2000,0x006e480650425c00,
0x0020380c50261000,0x005446500e406e00,
0x0000341c12340400,0x006a4a6208427a00,
0x0008640a201c0400,0x007602501a626a00,
0x000432103c140000,0x007a006a026a6a00,
0x000432103c340000,0x007a006a024a6a00,
0x00002c1858241000,0x00565266004a6e00,
0x0008680834240000,0x00760456484a6e00,
0x0000341e10340400,0x006a4a600a427a00,
0x000034380a640800,0x002a4a4650027600,
0x0008620c34240000,0x00760452484a6e00,
0x0000243e10340400,0x007652006a027a00,
0x00041c1012340800,0x006a622a68027600,
0x00106608382c0000,0x006e005046525600,
0x000826103c240000,0x0076106a02527600,
0x00202c08700c0000,0x005e42500e725600,
0x001016301c340000,0x006e204a224a6a00,
0x00201854082c2000,0x0054660856405e00,
0x0000241c482c2000,0x006e4a6016405e00,
0x001066100c140000,0x006e006a326a2a00,
0x00102c501c240000,0x006e400e604a6e00,
0x0000043c12240800,0x006a7a0268127600,
0x000414300a640800,0x002a620a50027600,
0x00202c0c50261000,0x005e42500a406e00,
0x0000243c08641000,0x006e4a4056086e00,
0x00002c38184c2000,0x0056524640225e00,
0x000434123a040400,0x007a0268007a2a00,
0x00202c0c50261000,0x005c42500a406e00,
0x00203c48482c2000,0x0056421610425e00,
0x00203c0c50261000,0x005642500a406e00,
0x0008241a38140000,0x00761260066a2a00,
0x0000283c086c2000,0x0056564056005e00,
0x0004082a12340400,0x002a761068027a00,
0x00202c48102c2000,0x005e40164c525600,
0x0000140c30161000,0x006a6a324a206e00,
0x0008241a38040000,0x00761260067a2a00,
0x0020380c54261000,0x005646500a406e00,
0x00102c48382c0000,0x006e421046525600,
0x001846303c240000,0x0066204a02527600,
0x000024381a240800,0x0076520660127600,
0x0020380444261000,0x005646581a406e00,
0x00202c087c100000,0x005e4056006e5600,
0x0008640a38340000,0x00760250064a6a00,
0x00204c083c2c0000,0x005e005640525600,
0x000024380a640800,0x002a5a4650027600,
0x00002c2c50261000,0x005652520a406e00,
0x00002c38084c2000,0x0056524650225e00,
0x0000043c10660800,0x002a7a026a007600,
0x000434101a300400,0x007a420a604e6a00,
0x0000243c04661800,0x006e4a405a006600,
0x0000301e10340400,0x006a4e206a027a00,
0x0000141c10360400,0x002a6a226a007a00,
0x000434103a040000,0x007a026a047a2a00,
0x00086402203c0400,0x007602585a422a00,
0x00086608382c0000,0x0076005644525600,
0x00202c48582c2000,0x005e421006525600,
0x000434121a100400,0x007a0268206e6a00,
0x000432103c240000,0x007a006a025a6a00,
0x000864120c140000,0x00760268326a2a00,
0x0020280444261000,0x005446581a406e00,
0x00202c085c040000,0x005e4056207a5600,
0x00202c500c2c2000,0x005e400e50425600,
0x000034380a640800,0x006a4a4650027600,
0x0000241c482c1000,0x006e4a6016406e00,
0x00043c1012340800,0x006a426a08427600,
0x001066300c140000,0x006e004a326a2a00,
0x001046102c240000,0x006e206a125a6a00,
0x0010662834140000,0x006e00520a6a2a00,
0x00201c48482c2000,0x0056621610425e00,
0x0000243c0c661000,0x006e4a4052006e00,
0x000866101c340000,0x0076004a624a6a00,
0x00204c083c240000,0x005e0056405a5600,
0x0004243a12340400,0x006a520068027a00,
0x0000243c48261000,0x006e4a4016406e00,
0x0020280454261000,0x005446580a406e00,
0x00202c08580c2000,0x005e425006725600,
0x000864323c040000,0x00760248027a2a00,
0x0000203408621800,0x00565e4856046600,
0x0020380444261000,0x005446581a406e00,
0x0004381212340400,0x006a466808427a00,
0x000434101a240800,0x006a4a6a00527600,
0x0008640a20340400,0x007602505a4a2a00,
0x0008640a30240000,0x007602504e5a6a00,
0x0008642a38040000,0x00760250067a2a00,
0x00202c085c200000,0x005e4056205e5400,
0x000864323c140000,0x00760248026a2a00,
0x000864121c140000,0x00760268226a2a00,
0x0000343448261000,0x002a4a4a12406e00,
0x0018660830280000,0x006600564c565600,
0x000866103c040000,0x0076006a027a2a00,
0x0000243c10361000,0x007652026a006e00,
0x000034300a640800,0x002a4a4e50027600,
0x000836101c340000,0x0076006a224a6a00,
0x001026483c240000,0x006e4016404a6e00,
0x00202c48182c0000,0x005e421066525600,
0x0010660c34240000,0x006e0052484a6e00,
0x0004341238140000,0x007a0268066a2a00,
0x0000283808661800,0x0056564456006600,
0x00002c2c10661000,0x005652504a006e00,
0x0008640a30340400,0x007602500a427a00,
0x00202c481c242000,0x005e4016404a5600,
0x001026101c140000,0x006e106a226a2a00,
0x00202c48382c0000,0x005e421046525600,
0x0020280450261000,0x005446580e406e00,
0x000024380a240800,0x0076520670127600,
0x000864322c040000,0x00760248127a2a00,
0x001026500c2c2000,0x006e400e50425c00,
0x00102458282c0000,0x006e4a0056525600,
0x0000341c18320400,0x006a4a6202447a00,
0x00102c483c240000,0x006e4016404a6e00,
0x00202c085c200000,0x005e4056205e5600,
0x001846302c240000,0x0066204a12527600,
0x001064182c2c0000,0x006e084650525600,
0x0000343408661000,0x002a4a4a52006e00,
0x0010261814140000,0x006e10622a6a2a00,
0x000864122c140000,0x00760268126a2a00,
0x00202c085c242000,0x005e4056004a5600,
0x00202c485c102000,0x005e4016006e5400,
0x0008640a30340400,0x007602700a427a00,
0x000434123a2c0400,0x007a026800526a00,
0x00041c201a240800,0x006a621a60127600,
0x00202c105c200000,0x005e404e205e5400,
0x00202c0850241000,0x005e42500e486e00,
0x001024483c240000,0x006e4816404a6e00,
0x0000340c10461000,0x006a4a326a206e00,
0x00043c1812340800,0x006a426208427600,
0x0010660834340000,0x006e00524a4a6a00,
0x00002c3848241000,0x00565246104a6e00,
0x00202c50082c2000,0x0056520e50425e00,
0x0000243c08661000,0x006e4a4056006e00,
0x000034340a640800,0x002a4a4a50027600,
0x00106c102c280000,0x006e004e50565400,
0x0000203408620800,0x00545e4856047600,
0x001066101c340000,0x006e000a624a6a00,
0x001026500c2c2000,0x006e400a52525400,
0x00201844102c2000,0x005466184e405e00,
0x0008241a10340400,0x007652600a427a00,
0x0018620c3c240000,0x00660452404a6e00,
0x0000243c30461800,0x007652024a206600,
0x0010264838280000,0x006e401644565600,
0x00002838084c2000,0x0054564456205e00,
0x00202c0858241000,0x005e4250064a6e00,
0x0020184c482c2000,0x0054661016405e00,
0x00204c183c240000,0x005e2046404a6e00,
0x00202c104c2c1000,0x005e404e10426e00,
0x0010245c082c2000,0x006c4a0056405e00,
0x0000283808660800,0x0056564456007600,
0x0004341238040000,0x007a0268067a6a00,
0x00102650042c2000,0x006e400e58425c00,
0x0004182a12340400,0x002a661068027a00,
0x00106418282c0000,0x006e084654525600,
0x0008642a34140000,0x007602500a6a2a00,
0x00102650043c2000,0x006e400a5a425400,
0x0000241c502c1000,0x006e4a600e406e00,
0x0020280c50261000,0x005446500e406e00,
0x00002c38086c1000,0x0056524456006e00,
0x0000241c50241000,0x006e4a600e486e00,
0x0010160834140000,0x006e20720a6a2a00,
0x0008640a30340400,0x007602504a4a6a00,
0x000866102c240000,0x0076006a12527600,
0x0000341c10661800,0x006a4a620a006600,
0x0000243408661000,0x006e4a4856006e00,
0x0000241c50261000,0x005a5a620a406e00,
0x0008240a38040000,0x00761270067a2a00,
0x000866303c140000,0x0076004a026a2a00,
0x001846300c140000,0x0066204a326a2a00,
0x000034300a640800,0x006a4a4e50027600,
0x001026542c280000,0x006e400a50565400,
0x0000241c50261000,0x002a5a620a406e00,
0x000866102c040000,0x0076006a127a2a00,
0x000432181c140000,0x007a0462226a2a00,
0x00001074082c2000,0x00546e0856405e00,
0x0020280450261000,0x005656580e406e00,
0x00202c481c242000,0x005e4016404a5c00,
0x00002c300c680800,0x0056524c52047600,
0x00002c18482c2000,0x0056526610425e00,
0x0008640a20340400,0x007602501a427a00,
0x00202c483c240000,0x005e4016404a6e00,
0x0004243a12340400,0x0072520068027a00,
0x000434121c140000,0x007a0268226a6a00,
0x00202c0444261000,0x005c42581a406e00,
0x00202c18482c1000,0x0056524610426e00,
0x0010264834240000,0x006e4016484a6e00,
0x001024500c2c2000,0x006e480e50425c00,
0x0000103608340400,0x006a6e0872027a00,
0x0008642a30340400,0x007602504a027a00,
0x00002c38086c2000,0x0056524650025e00,
0x000414302a640800,0x002a620a50027600,
0x001024501c240000,0x006e480e604a6e00,
0x0000243c32640800,0x0076520248027600,
0x000028340c661800,0x0056564852006600,
0x0010462834140000,0x006e20520a6a2a00,
0x001846102c240000,0x0066206a12527600,
0x0000241c50261000,0x00565a620a406e00,
0x0000141c30161000,0x002a6a224a206e00,
0x00203c50482c2000,0x0056420e10425e00,
0x0010265404282000,0x006e400a58465600,
0x0020245c082c2000,0x005c4a0056405e00,
0x0000341c10361000,0x006a4a620a406e00,
0x0008240a10340400,0x007612700a427a00,
0x0008640a38240000,0x00760250465a6a00,
0x00102458082c2000,0x006e480654425600,
0x000866083c240000,0x00760056404a6e00,
0x001066083c240000,0x006e0056404a6e00,
0x0000242c10660800,0x007652126a007600,
0x0008640a20340400,0x007602701a427a00,
0x0008640a38240000,0x0076025006527600,
0x001862083c240000,0x00660456404a6e00,
0x0000283c0c621800,0x0056564052046600,
0x000434101e300000,0x007a026a204e6a00,
0x00202c500c282000,0x005e400e50465600,
0x00202c500c2c2000,0x005e400e50425c00,
0x000024380a640800,0x0076520650027600,
0x00202c10482c1000,0x0056524e10426e00,
0x0008640a30340000,0x007602504e4a6a00,
0x00202c0874140000,0x005e4056086a5600,
0x0010264404282000,0x006e401a58465600,
0x00203c0858241000,0x00564256004a6e00,
0x00002838086c0800,0x0056564456007600,
0x00200c50482c2000,0x0056720c16405e00,
0x00106c0838280000,0x006e005644565400,
0x0004341238040000,0x007a0268067a2a00,
0x0000107c082c2000,0x00566e0056405e00,
0x001846103c240000,0x0066206a02527600,
0x001026500c2c2000,0x006e400a52425e00,
0x000414300a640800,0x002a6a0a50027600,
0x000434200a640800,0x007a421a70027600,
0x0004301a12340400,0x006a4e6008427a00,
0x0004142022640800,0x002a621a58027600,
0x000024380a640800,0x005a5a4650027600,
0x0008241a14340000,0x007612602a4a6a00,
0x0008642a38140000,0x00760250066a2a00,
0x001046203c240000,0x006e205a02527600,
0x0000243c084c2000,0x006e4a4056205e00,
0x0010462824140000,0x006e20521a6a2a00,
0x001026500c2c2000,0x006e400a50525600,
0x00102c501c200000,0x006e400e605e5600,
0x001066101c340000,0x006e004a624a6a00,
0x0010264c38280000,0x006e401244565400,
0x001866103c240000,0x0066006a02527600,
0x0010160834140000,0x006e20720a6a6a00,
0x00203c0450261000,0x005642580a406e00,
0x0008241a10340400,0x007612606a027a00,
0x000434320a340400,0x006a420870027a00,
0x0008640a201c0400,0x007602501a622a00,
0x00202c087c080000,0x005e405600765600,
0x00002818482c1000,0x0054566416406e00,
0x00043c300a640800,0x006a420a50027600,
0x000434200a640800,0x003a421a50027600,
0x00043c302a640800,0x006a420a50027600,
0x0008640a301c0400,0x007602700a622a00,
0x00002c0c50261000,0x005652700a406e00,
0x0000240c50261000,0x00565a720a406e00,
0x0020280c54261000,0x005656500a406e00,
0x001866101c340000,0x0066004a224a6a00,
0x00002c3848261000,0x0056524412406e00,
0x001026500c3c2000,0x006e400a52425600,
0x00086608382c0000,0x0076005046525600,
0x0004340a321c0400,0x007a027008626a00,
0x000024300a640800,0x006a5a4e50027600,
0x0000243448261000,0x006e4a4816406e00,
0x00201c48082c2000,0x0056621650425e00,
0x000024340c661000,0x006e4a4852006e00,
0x0004083a10340400,0x002a76006a027a00,
0x00102c481c242000,0x006e4016404a5600,
0x000434121a140400,0x007a0268206a2a00,
0x00102c500c2c2000,0x006e400e50425600,
0x0008340a301c0400,0x007602700a626a00,
0x0000203a10340400,0x006a5e006a027a00,
0x0018460834140000,0x006620720a6a6a00,
0x000434103e300000,0x007a026a004e6a00,
0x0008660834140000,0x007600520a6a6a00,
0x000028340c661800,0x0054564852006600,
0x000864323c240000,0x0076024802527600,
0x00202c106c080000,0x005e404e10765600,
0x0008640a30140400,0x007602700a6a2a00,
0x00106c083c280000,0x006e005640565400,
0x0004340832340800,0x007a027208427600,
0x0008640a38140000,0x00760250066a6a00,
0x00202c50482c2000,0x0056520e10425e00,
0x0004243812340800,0x006a520268027600,
0x0000341418261000,0x006a4a6a02506e00,
0x00202c0850241000,0x005e42500e4a6e00,
0x000864123c040000,0x00760268027a2a00,
0x0010264834280000,0x006e401648565600,
0x000414300a640800,0x006a6a0a50027600,
0x0000243c04661000,0x006e4a405a006e00,
0x00043c1210340400,0x006a42680a427a00,
0x000024380a640800,0x004a5a4650027600,
0x000864223c140000,0x00760258026a2a00,
0x0008640a203c0400,0x007602505a425a00,
0x000414300a640800,0x002a620a70027600,
0x000866043c240000,0x0076005a404a6e00,
0x0020380c54261000,0x005446500a406e00,
0x000434123c040000,0x007a0268027a2a00,
0x00102644042c2000,0x006e401a58425600,
0x0010660834240000,0x006e0056484a6e00,
0x000864122c040000,0x00760268127a2a00,
0x001026500c2c2000,0x006e400e50425600,
0x001026500c280000,0x006e400a70565400,
0x00202c0c54261000,0x005c42500a406e00,
0x000438320a340400,0x006a460870027a00,
0x00041c200a640800,0x006a621a50027600,
0x0008680c38280000,0x0076045244565400,
0x00203c50082c2000,0x0056420e50425e00,
0x00202c0858241000,0x00565256004a6e00,
0x0000341c10261000,0x006a4a620a506e00,
0x00102648382c0000,0x006e401244525600,
0x00202c4818280000,0x005e401664565400,
0x00102458082c2000,0x006e4a0056425e00,
0x001024580c2c2000,0x006e480650425600,
0x00203c18482c2000,0x0056424610425e00,
0x0000341812340800,0x006a4a6608427600,
0x00002c3808641000,0x00565246500a6e00,
0x000434101a240800,0x007a420a60127600,
0x0004243812340800,0x0072520268027600,
0x0008642a38240000,0x0076025006527600,
0x0018620834200000,0x00660456485e5400,
0x0018460824140000,0x006620721a6a6a00,
0x00186608382c0000,0x0066005246525600,
0x00202c087c0c0000,0x005e405600725600,
0x0020280454261000,0x005456580a406e00,
0x0008640a203c0400,0x007602505a426a00,
0x00043c3a12340400,0x006a420068027a00,
0x000864322c240000,0x0076024812527600,
0x0020380440261000,0x005646581a406e00,
0x001046303c140000,0x006e204a026a2a00,
0x000862142c280000,0x0076044a50565400,
0x00001074082c2000,0x00566e0856405e00,
0x00202c104c2c2000,0x005e404e10425600,
0x0010660c3c240000,0x006e0052404a6e00,
0x0008341218140000,0x00760268266a2a00,
0x00204c083c240000,0x005e2056404a6e00,
0x00102650042c2000,0x006e400a58425e00,
0x0000045c082c2000,0x00567a0056405e00,
0x000434103e140000,0x007a026a006a6a00,
0x0008660830280000,0x007600564c565600,
0x00202c504c2c2000,0x005e400e10425e00,
0x000432181c340000,0x007a4402624a6a00,
0x0020380c44261000,0x005446501a406e00,
0x000434101a240800,0x007a420a60527600,
0x0020381c482c1000,0x0056464016406e00,
0x0000243c12240800,0x0076520268127600,
0x001046303c240000,0x006e204a02527600,
0x001024580c282000,0x006e480650465600,
0x0018662834140000,0x006600520a6a2a00,
0x001026541c280000,0x006e400a60565400,
0x0000143408161000,0x002a6a0a72206e00,
0x0008640a30340400,0x007602500a4a6a00,
0x00202c0c50241000,0x005e42500e486e00,
0x000034300a640800,0x006a4a0e50027600,
0x0000043a10340400,0x006a7a006a027a00,
0x0008660c3c240000,0x00760052404a6e00,
0x0000283c08680800,0x0056564056047600,
0x00204c1838280000,0x005e204644565600,
0x0000243c20661000,0x007652025a006e00,
0x0004243812340400,0x0072520268027a00,
0x00041c300a640800,0x006a620a50027600,
0x0000205c082c2000,0x00565e0056405e00,
0x00202c4850082000,0x005e40160c765400,
0x0000245c082c2000,0x00565a0056405e00,
0x0020380450261000,0x005446580e406e00,
0x00202c58482c2000,0x0056520610425e00,
0x00041c200a640800,0x002a621a50027600,
0x0000086c102c2000,0x005676104e405e00,
0x001066102c2c0000,0x006e004a52525600,
0x0020241c482c1000,0x005c4a4016406e00,
0x0020380858241000,0x0056465406486e00,
0x0000087c082c2000,0x0054760056405e00,
0x0008241a10340400,0x007652006a027a00,
0x001866303c240000,0x0066004a02527600,
0x000434101a240800,0x007a426a00527600,
0x0010460824140000,0x006e20721a6a2a00,
0x00203c08482c1000,0x0056425610426e00,
0x000434100a240800,0x007a426a10527600,
0x0020245c082c2000,0x00564a0056405e00,
0x0008642a30340400,0x007602500a427a00,
0x001024500c282000,0x006e480e50465600,
0x0004183a12340400,0x006a660068027a00,
0x00202c0858241000,0x0056425406486e00,
0x00002c3808660800,0x0056524452007600,
0x0008360834140000,0x007600720a6a2a00,
0x001866102c2c0000,0x0066004a50525600,
0x00002c38086c1000,0x0056524650026e00,
0x0020380454261000,0x005446580a406e00,
0x0000283048261000,0x0054564c16406e00,
0x0020380858241000,0x0054465406486e00,
0x001866302c240000,0x0066004a12527600,
0x0020284c502c2000,0x005646100e405e00,
0x0008341238140000,0x00760268066a2a00,
0x0020280c54261000,0x005456500a406e00,
0x00202c0454261000,0x005c42580a406e00,
0x000864121c140000,0x00760268226a6a00,
0x000434103a200000,0x007a026a005e6a00,
0x0010265004282000,0x006e400e58465600,
0x000014340a640800,0x006a6a0a50027600,
0x000434300a640800,0x006a4a0a70027600,
0x0020241c482c1000,0x00564a4016406e00,
0x0000243c08680800,0x00565a4056047600,
0x0000281c54261000,0x005456600a406e00,
0x0004340a301c0400,0x007a02700a626a00,
0x001066043c240000,0x006e005a404a6e00,
0x0000082e10340400,0x002a76106a027a00,
0x0008620834240000,0x00760456484a6e00,
0x000834121c140000,0x00760268226a2a00,
0x0008642a20340400,0x007602501a427a00,
0x0008660834240000,0x00760056484a6e00,
0x0000207c082c2000,0x00545e0056405e00,
0x0008640a20140400,0x007602501a6a6a00,
0x0000243c0c661800,0x00565a4052006600,
0x00202c4818280000,0x005e401664565600,
0x001024500c2c2000,0x006e480e50425600,
0x0000143418260800,0x006a6a0a62107600,
0x00002c3808661000,0x0056524452006e00,
0x0020205c482c2000,0x00545e0016405e00,
0x000034141a240800,0x006a4a6a00527600,
0x001866303c140000,0x0066004a026a2a00,
0x0008640a34140000,0x007602500a6a6a00,
0x0000281c502c2000,0x005456600e405e00,
0x00202c0858241000,0x005e4056044a6e00,
0x00202c0858241000,0x005c425406486e00,
0x0000282c10660800,0x005656504a007600,
0x00203858482c2000,0x0056460416405e00,
0x0020045c482c2000,0x00567a0016405e00,
0x0020145c082c2000,0x00566a0056405e00,
0x0000141c12340400,0x006a6a2268027a00,
0x00002c1c50261000,0x005652600a406e00,
0x0000286c102c2000,0x005456104e405e00,
0x00102650042c2000,0x006e400e58425600,
0x0000203c184c2000,0x00545e4046205e00,
0x0008640a30340000,0x007602700e4a6a00,
0x00204c103c240000,0x005e204e404a6e00,
0x00202c105c200000,0x005e404e205e5600,
0x001046103c240000,0x006e206a02527600,
0x001866203c140000,0x0066005a026a2a00,
0x000864223c040000,0x00760258027a2a00,
0x00102c500c282000,0x006e400e50465600,
0x00086402201c0400,0x007602581a622a00,
0x0008642a301c0400,0x007602500a622a00,
0x0018660834140000,0x006600720a6a2a00,
0x001046302c240000,0x006e204a12527600,
0x0008640a38240000,0x0076027006527600,
0x00203c58082c2000,0x0056420650425e00,
0x000868102c280000,0x0076044e50565400,
0x00202c0858241000,0x005e405406486e00,
0x00001474082c2000,0x00566a0856405e00,
0x0010460834140000,0x006e20720a6a6a00,
0x000432183c040000,0x007a0462027a2a00,
0x0000243c22640800,0x0076520258027600,
0x001016103c040000,0x006e206a027a2a00,
0x000434120a100400,0x007a0268306e2a00,
0x0020380c50261000,0x005646500e406e00,
0x0000203c0c621800,0x00545e4052046600,
0x0008241a18340000,0x00765200664a6a00,
0x001846300c140000,0x0066204a326a6a00,
0x00200858082c2000,0x0054760456405e00,
0x00206c083c240000,0x005e0056404a6e00,
0x000430320a340400,0x006a4e0870027a00,
0x00202c0454261000,0x005e42580a406e00,
0x001026540c2c2000,0x006e400a50425e00,
0x0000282818640800,0x0054565446087600,
0x0008642a30140400,0x007602500a6a2a00,
0x00202c485c242000,0x005e4016004a5e00,
0x00002c3808621800,0x0056524456046600,
0x0018660834240000,0x00660056484a6e00,
0x00102640043c2000,0x006e401a5a425600,
0x0004340a38040000,0x007a0270067a2a00,
0x00186208382c0000,0x0066045644525600,
0x00202c4c502c2000,0x005e42100e405e00,
0x001046203c040000,0x006e205a027a2a00,
0x0008642220340400,0x007602581a427a00,
0x0020280450261000,0x005646580a406e00,
0x0020284c102c2000,0x005646104e405e00,
0x00102458082c2000,0x006e480456405e00,
0x00202c0878280000,0x005e405604565600,
0x0000243812340400,0x0076520668027a00,
0x001026500c2c2000,0x006e400a52525600,
0x0008642a301c0400,0x007602500a626a00,
0x00202c485c242000,0x005e4016005a5600,
0x00102650042c2000,0x006e400a5a525400,
0x00202c0c50261000,0x005642500a406e00,
0x0008640a30140000,0x007602700e6a2a00,
0x00202c0c50261000,0x005652500e406e00,
0x00102c48182c2000,0x006e421046525600,
0x000414320a340400,0x006a620870027a00,
0x0010264c30280000,0x006e40124c565400,
0x000432101c140000,0x007a046a226a2a00,
0x00002c2c502c1000,0x005652500e406e00,
0x000866101c340000,0x0076004a224a6a00,
0x000414101a240800,0x002a622a60127600,
0x0020085c482c2000,0x0054760016405e00,
0x00002c3008621800,0x0056524c56046600,
0x0004341812340800,0x006a4a6208427600,
0x00200c58082c2000,0x0056720456405e00,
0x00202c18482c2000,0x0056524610425e00,
0x00002c38086c0800,0x0056524456007600,
0x000434121a1c0400,0x007a026820622a00,
0x00002c6c102c2000,0x005652104e405e00,
0x00203c0c58241000,0x0056425006486e00,
0x00202c4848182000,0x005e401614665600,
0x0000283448261000,0x0054564816406e00,
0x00202c48580c2000,0x005e421006725600,
0x000866301c140000,0x0076004a226a6a00,
0x00041c300a640800,0x006a620a70027600,
0x00086608382c0000,0x0076005246525600,
0x00086422301c0400,0x007602580a622a00,
0x00201c50482c2000,0x0056620c16405e00,
0x001066203c240000,0x006e005a02527600,
0x00102458082c2000,0x006e4a0456405e00,
0x0000341c10320400,0x006a4a620a447a00,
0x000434123c140000,0x007a0268026a2a00,
0x00102640043c2000,0x006e401a5a425400,
0x0000241c50261000,0x00565a600a406e00,
0x00086418282c0000,0x00760a4056525600,
0x00086402203c0400,0x007602585a425a00,
0x0020241c482c1000,0x005e4a4016406e00,
0x001866300c140000,0x0066004a326a2a00,
0x000034101a240800,0x006a4a6e00527600,
0x0020241c482c2000,0x005c4a4016405e00,
0x0020305c482c2000,0x00544e0016405e00,
0x00202c104c2c2000,0x005e404e10425e00,
0x0000043c10161000,0x006a7a026a206e00,
0x0008640a38040000,0x00760250067a6a00,
0x00203c48082c2000,0x0056421650425e00,
0x00002c0c58241000,0x0056527006486e00,
0x00202c4858082000,0x005e401604765400,
0x00043c0812340800,0x006a427208427600,
0x00043c202a640800,0x006a421a50027600,
0x0020280c54261000,0x005646500a406e00,
0x00043c1812340400,0x006a426208427a00,
0x000414301a240800,0x002a620a60127600,
0x00202c10482c2000,0x0056524e10425e00,
0x00002c2c106c2000,0x005652504e005e00,
0x00202844102c2000,0x005656184e405e00,
0x00204c1838280000,0x005e204644565400,
0x0000243c10661800,0x007652026a006600,
0x000864223c240000,0x0076025802527600,
0x0004340836140000,0x007a0272086a6a00,
0x001026500c2c2000,0x006e400e50425e00,
0x00201c48482c2000,0x0056621416405e00,
0x00203c10482c2000,0x0056424e10425e00,
0x00106c083c200000,0x006e0056405e5400,
0x00086c0838280000,0x0076005644565400,
0x0018660834140000,0x006600520a6a6a00,
0x0000343408661800,0x006a4a0a52006600,
0x0000243c12340400,0x0076520268027a00,
0x000014342a640800,0x006a6a0a50027600,
0x0020380458241000,0x0056465806486e00,
0x0004340832180400,0x007a027208662a00,
0x0004340a38240000,0x007a027006527600,
0x00203c08482c2000,0x0056425610425e00,
0x0018660c38280000,0x0066005244565400,
0x0008240a38140000,0x00761270066a2a00,
0x0010265404382000,0x006e400a58465400,
0x00102654042c2000,0x006e400a58425e00,
0x0010264838280000,0x006e401644565400,
0x000866102c2c0000,0x0076004a52525600,
0x00086402203c0400,0x007602585a426a00,
0x00202c4c502c2000,0x005642100e405e00,
0x001026540c282000,0x006e400a50565400,
0x00202c485c042000,0x005e4016007a5600,
0x0000203c184c2000,0x00565e4046205e00,
0x000414300a640800,0x006a620a50027600,
0x0018660838280000,0x0066005644565400,
0x00202c0c50241000,0x005c42500e486e00,
0x0008341218340000,0x00760268264a6a00,
0x0008640a303c0400,0x007602504a425a00,
0x00201c58082c2000,0x0056620456405e00,
0x00202c0c54261000,0x005e42500a406e00,
0x0020380450241000,0x005646580e486e00,
0x0000103e10340400,0x002a6e006a027a00,
0x000024382a640800,0x0076520650027600,
0x000414200a640800,0x006a6a1a50027600,
0x000824123c140000,0x00761268026a2a00,
0x0008642a201c0400,0x007602501a622a00,
0x000434123a240400,0x007a026800527a00,
0x001862083c280000,0x0066045640565400,
0x00086c083c200000,0x00760056405e5400,
0x00002838184c2000,0x0054564446205e00,
0x000028340c661000,0x0054564852006e00,
0x000866303c240000,0x0076004a02527600,
0x0000042c30660800,0x002a7a124a007600,
0x0000340c30161000,0x006a4a324a206e00,
0x0008240a10340400,0x007652700a427a00,
0x0010265004382000,0x006e400a58465400,
0x001046302c140000,0x006e204a126a2a00,
0x0020280c50261000,0x005656500e406e00,
0x0000203c0c661800,0x00565e4052006600,
0x0008642a38140000,0x00760250066a6a00,
0x00002c2c10621800,0x005652504e046600,
0x0000242c10461800,0x007652126a206600,
0x000414300a240800,0x002a620a70127600,
0x00102c4c102c2000,0x006e42104e405e00,
0x00002838086c0800,0x0054564456007600,
0x001846102c040000,0x0066206a127a2a00,
0x0000141e10340400,0x002a6a206a027a00,
0x00043c2022640800,0x006a421a58027600,
0x0018462824140000,0x006620521a6a2a00,
0x001024580c2c2000,0x006e480650425e00,
0x0020201c482c1000,0x00545e4016406e00,
0x0004341222180400,0x007a026818662a00,
0x000866203c240000,0x0076005a02527600,
0x0008241a10340400,0x007652006a427a00,
0x0000282c50241000,0x005656500e486e00,
0x00202c0c502c2000,0x005c42500e405e00,
0x00202c106c2c0000,0x005e404e10525600,
0x0008641828280000,0x0076084654565400,
0x00102650043c2000,0x006e400a5a425600,
0x0000140c30461000,0x002a6a324a206e00,
0x00041c101a240800,0x002a622a60127600,
0x0000341c12340800,0x006a4a6208427600,
0x00041c320a340400,0x006a620870027a00,
0x000834123c240000,0x0076026802527600,
0x000864123c240000,0x0076026802527600,
0x00002c3808661000,0x0056524456006e00,
0x00102c4818280000,0x006e401664565400,
0x0008640a34140000,0x007602700a6a2a00,
0x00202c4858282000,0x005e401604565400,
0x00106608382c0000,0x006e005246525600,
0x000864122c240000,0x0076026812527600,
0x0020201c482c1000,0x00565e4016406e00,
0x0000143408661800,0x006a6a0a52006600,
0x00203844102c2000,0x005646184e405e00,
0x0008660834340000,0x007600520a4a6a00,
0x0004381a10340400,0x006a46600a427a00,
0x001866102c240000,0x0066006a12527600,
0x000434103a240000,0x007a026a005a6a00,
0x00106c083c200000,0x006e0056405e5600,
0x00202c085c102000,0x005e4056006e5400,
0x0004321814340000,0x007a44026a4a6a00,
0x00202c0c58241000,0x005c425006486e00,
0x00041c2002640800,0x002a621a58027600,
0x00043412083c0400,0x007a420872426a00,
0x00106c102c280000,0x006e004e50565600,
0x0000201c50261000,0x00565e600a406e00,
0x0000243c10360400,0x007652026a007a00,
0x00002838482c1000,0x0054564416406e00,
0x00202c0c502c1000,0x005c42500e406e00,
0x00202c105c240000,0x005e404e205a5600,
0x001036103c040000,0x006e006a027a2a00,
0x00086422201c0400,0x007602581a622a00,
0x0000341c12640800,0x006a4a2248027600,
0x0018660c3c200000,0x00660052405e5400,
0x001846303c040000,0x0066204a027a2a00,
0x00202c08581c2000,0x005e425006625600,
0x0004380a12340400,0x006a467008427a00,
0x000824123c040000,0x00761268027a2a00,
0x000434101a240800,0x007a026a20127600,
0x00202c0450261000,0x005652580e406e00,
0x00202c4850182000,0x005e40160c665400,
0x001866100c140000,0x0066006a326a2a00,
0x0000343408661000,0x006a4a0a52006e00,
0x0004082a12340400,0x006a761068027a00,
0x0020245c482c2000,0x005e4a0016405e00,
0x00102c4838280000,0x006e401644565400,
0x0008640a38240000,0x00760250065a6a00,
0x000866203c040000,0x0076005a027a2a00,
0x00043c1212340400,0x006a426808427a00,
0x0000143c12340400,0x002a6a0268027a00,
0x00202c481c382000,0x005e401640465600,
0x00202c0c54261000,0x005642500a406e00,
0x000834123c040000,0x00760268027a2a00,
0x0000201c482c1000,0x00545e6016406e00,
0x00202c0444261000,0x005e42581a406e00,
0x0000143408661000,0x002a6a0a52006e00,
0x001066302c140000,0x006e004a126a2a00,
0x0010265004282000,0x006e400a58565400,
0x00202c08782c0000,0x005e425006525600,
0x0008640a301c0400,0x007602500a626a00,
0x001066101c140000,0x006e006a226a2a00,
0x0008642a20140400,0x007602501a6a2a00,
0x0020380450261000,0x005646580e406e00,
0x0008340a38240000,0x0076027006527600,
0x0008640a24140000,0x007602701a6a2a00,
0x000864321c140000,0x00760248226a6a00,
0x0000243c482c2000,0x00565a4016405e00,
0x0000207c082c2000,0x00565e0056405e00,
0x0020245c482c2000,0x00564a0016405e00,
0x00043c1a10340400,0x006a42600a427a00,
0x0000143812340400,0x002a6a0668027a00,
0x00203014482c1000,0x00544e4816406e00,
0x000436101c140000,0x007a006a226a2a00,
0x0004340a38140000,0x007a0270066a2a00,
0x000434122a040400,0x007a0268107a6a00,
0x0010660c3c200000,0x006e0052405e5600,
0x0000242c10661800,0x007652126a006600,
0x0004341208140400,0x007a0268326a6a00,
0x00204c18382c0000,0x005e224046525600,
0x0010660834340000,0x006e00520a4a6a00,
0x00203818482c1000,0x0054464416406e00,
0x000024380a340400,0x0076520670027a00,
0x001846301c140000,0x0066204a226a6a00,
0x00106608382c0000,0x006e005244525600,
0x0000341c10360400,0x006a4a620a407a00,
0x00202c485c3c2000,0x005e401600425600,
0x0018660824140000,0x006600721a6a6a00,
0x0018660834200000,0x00660056485e5400,
0x0008642a28140000,0x00760250166a2a00,
0x000824123c240000,0x0076126802527600,
0x00102458082c2000,0x006e480654425e00,
0x00202c0c50261000,0x005c42500e406e00,
0x00102644042c2000,0x006e401a58425e00,
0x001036101c140000,0x006e006a226a2a00,
0x00202c0450261000,0x005c42580e406e00,
0x00043c300a640800,0x006a420a70027600,
0x0008640a303c0400,0x007602700a426a00,
0x000836101c140000,0x0076006a226a2a00,
0x0000242c30461800,0x007652124a206600,
0x0000043c12340400,0x002a7a0268027a00,
0x0000203c48261000,0x00545e4016406e00,
0x000866302c240000,0x0076004a12527600,
0x0004340a32340400,0x007a027008427a00,
0x0018460834140000,0x006620720a6a2a00,
0x0000341418320400,0x006a4a6a02447a00,
0x00203c04502c2000,0x005642580e405e00,
0x000432183c140000,0x007a0462026a2a00,
0x0004181a12340400,0x006a662068027a00,

	};

	std::set<std::array<uint64_t, 2>>answers;

	for (int i = 0; i < 3200; i += 2) {
		Board b(examples[i], examples[i + 1]);
		const auto m = get_moves(b.player, b.opponent);
		assert(_mm_popcnt_u64(m) >= 34);
		const auto uni = b.unique();
		answers.insert(uni);
	}
	return std::vector<std::array<uint64_t, 2>>(answers.begin(), answers.end());
}

int main() {

	{
		for (int i = 5; i <= 15; ++i) {
			searched.clear();
			leafnode.clear();
			DISCS = i;
			search(Board::initial());
			std::cout << "discs = " << i << ": internal nodes = " << searched.size() << ", leaf nodes = " << leafnode.size() << std::endl;
		}

		const auto examples = move34s();

		for (int i = 0; i < examples.size(); ++i) {
			Board bb(examples[i][0], examples[i][1]);
			retroflips.resize(bb.popcount() + 1);
			retrospective_searched.clear();
			std::cout << "start: " << i << std::endl;
			const bool x = retrospective_search(bb, false);
			if (x) {
				std::cout << "succeeded" << std::endl;
				return 0;
			}
		}
		std::cout << "failed" << std::endl;

		return 0;
	}

	{
		searched.clear();
		leafnode.clear();
		DISCS = 15;
		search(Board::initial());
		std::cout << "discs = " << 15 << ": internal nodes = " << searched.size() << ", leaf nodes = " << leafnode.size() << std::endl;

		const uint64_t bb_opponent =
			0b0000000001101110010000000000101001100010010110100101001000000000;
		const uint64_t bb_player =
			0b0000000000010000001001100101000000011100001001000000000000000000;

		Board bb(bb_player, bb_opponent);
		std::cout << board_print(bb) << std::endl;

		retroflips.resize(bb.popcount() + 1);
		retrospective_searched.clear();
		const bool x = retrospective_search(bb, false);

		std::cout << (x ? "succeeded" : "failed") << std::endl;
		return 0;
	}

	{
		std::array<uint64_t, 10000> tmp;

		int64_t b = ~0x0000001818000000ULL;

		int answer = 0, amax = 0;

		for (unsigned long index = 0; _BitScanForward64(&index, b); b &= b - 1) {

			const int x = retrospective_flip(index, 0, 0xFFFF'FFFF'FFFF'FFFFULL, tmp);
			answer += x;
			if (amax < x)amax = x;
		}

		std::cout << answer << std::endl;
		std::cout << amax << std::endl;
	}

	Test__IsConnected();

	return 0;
}




