
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
#include<unordered_set>
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
#include<chrono>
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

namespace prime_world {

/*
 * https://github.com/primenumber/issen/blob/72f450256878094ffe90b75f8674599e6869c238/src/move_generator.cpp
 *
 * This source code is licensed under the
 * GNU General Public License v3.0
 */

inline __m256i flipVertical(__m256i dbd) {
	return __m256i(_mm256_shuffle_epi8(dbd, _mm256_set_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7)));
}
inline __m256i upper_bit(__m256i p) {
	p = _mm256_or_si256(p, _mm256_srli_epi64(p, 1));
	p = _mm256_or_si256(p, _mm256_srli_epi64(p, 2));
	p = _mm256_or_si256(p, _mm256_srli_epi64(p, 4));
	p = _mm256_andnot_si256(_mm256_srli_epi64(p, 1), p);
	p = flipVertical(p);
	p = _mm256_and_si256(p, _mm256_sub_epi64(_mm256_setzero_si256(), p));
	return flipVertical(p);
}
inline uint64_t hor(const __m256i lhs) {
	__m128i lhs_xz_yw = _mm_or_si128(_mm256_castsi256_si128(lhs), _mm256_extractf128_si256(lhs, 1));
	return uint64_t(_mm_cvtsi128_si64(lhs_xz_yw) | _mm_extract_epi64(lhs_xz_yw, 1));
}
uint64_t flip(uint64_t player, uint64_t opponent, int pos) {

	const __m256i pppp = _mm256_set1_epi64x(player);
	const __m256i oooo = _mm256_set1_epi64x(opponent);

	const __m256i OM = _mm256_and_si256(oooo, _mm256_set_epi64x(0xFFFFFFFFFFFFFFFFULL, 0x7E7E7E7E7E7E7E7EULL, 0x7E7E7E7E7E7E7E7EULL, 0x7E7E7E7E7E7E7E7EULL));

	__m256i flipped, outflank, mask;


	mask = _mm256_srli_epi64(_mm256_set_epi64x(0x0080808080808080ULL, 0x7F00000000000000ULL, 0x0102040810204000ULL, 0x0040201008040201ULL), 63 - pos);

	outflank = _mm256_and_si256(upper_bit(_mm256_andnot_si256(OM, mask)), pppp);

	flipped = _mm256_and_si256(_mm256_slli_epi64(_mm256_sub_epi64(_mm256_setzero_si256(), outflank), 1), mask);


	mask = _mm256_slli_epi64(_mm256_set_epi64x(0x0101010101010100ULL, 0x00000000000000FEULL, 0x0002040810204080ULL, 0x8040201008040200ULL), pos);

	outflank = _mm256_and_si256(_mm256_and_si256(mask, _mm256_add_epi64(_mm256_or_si256(OM, _mm256_andnot_si256(mask, _mm256_set1_epi8(0xFF))), _mm256_set1_epi64x(1))), pppp);

	flipped = _mm256_or_si256(flipped, _mm256_and_si256(_mm256_sub_epi64(outflank, _mm256_add_epi64(_mm256_cmpeq_epi64(outflank, _mm256_setzero_si256()), _mm256_set1_epi64x(1))), mask));

	return hor(flipped);
}

};

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

int DISCS = 10;

int pass_num_max = 0;

std::set<std::pair<int, std::array<uint64_t, 2>>>searched_pass;

void search_to_get_passmax(const Board &board, int pass_num) {

	const auto pop = board.popcount();

	if (pop >= DISCS) {
		if (get_moves(board.player, board.opponent)) {
			if (pass_num_max < pass_num)pass_num_max = pass_num;
			return;
		}
		else if (get_moves(board.opponent, board.player)) {
			Board next;
			next.player = board.opponent;
			next.opponent = board.player;
			search_to_get_passmax(next, pass_num + 1);
		}
		return;
	}

	if (pop < 16) {
		const auto uni = std::make_pair(pass_num, board.unique());
		if (searched_pass.find(uni) != searched_pass.end())return;
		searched_pass.insert(uni);
	}

	Board next;
	uint64_t b = get_moves(board.player, board.opponent);

	if (b == 0) {
		if (get_moves(board.opponent, board.player)) {
			next.player = board.opponent;
			next.opponent = board.player;
			search_to_get_passmax(next, pass_num + 1);
		}
		return;
	}

	for (unsigned long index = 0; _BitScanForward64(&index, b); b &= b - 1) {
		const uint64_t flipped = flip(index, board.player, board.opponent);
		if (flipped == 0) continue;
		next.player = board.opponent ^ flipped;
		next.opponent = board.player ^ (flipped | (1ULL << index));
		search_to_get_passmax(next, pass_num);
	}
}

int main() {

	for (int i = 15; i <= 17; ++i) {
		searched_pass.clear();
		DISCS = i;
		pass_num_max = 0;
		const auto start = std::chrono::time_point_cast<std::chrono::seconds>(std::chrono::system_clock::now()).time_since_epoch().count();
		search_to_get_passmax(Board::initial(), 0);
		const auto end = std::chrono::time_point_cast<std::chrono::seconds>(std::chrono::system_clock::now()).time_since_epoch().count();
		std::cout << "discs = " << i << ": internal nodes = " << searched_pass.size() << ", pass_num_max = " << pass_num_max << std::endl;
		std::cout << "elapsed time = " << (end - start) << " seconds" << std::endl;
	}

	return 0;
}

