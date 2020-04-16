# coding: UTF-8

# Author: Hiroki Takizawa, 2020

# License: MIT License

# Copyright(c) 2020 Hiroki Takizawa
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import time
import re
import sys
import pandas as pd

import z3
# z3をimportする簡単な方法：
# https://qiita.com/SatoshiTerasaki/items/476c9938479a4bfdda52


def solve(threshold):

    MAX_TURN = 42

    s = z3.Goal() # z3.Solver()
    
    bb_player = [z3.BitVec(f"bb_player_{i}", 64) for i in range(MAX_TURN+2)]
    bb_opponent = [z3.BitVec(f"bb_opponent_{i}", 64) for i in range(MAX_TURN+2)]
    pop_move = [z3.BitVec(f"pop_move_{i}", 64) for i in range(MAX_TURN+2)]

    directions1 = [z3.BitVec(f"directions1_{i}", 64) for i in range(4)]
    directions2 = [z3.BitVec(f"directions2_{i}", 64) for i in range(4)]

    flip_const1 = [z3.BitVec(f"flip_const1_{i}", 64) for i in range(4)]
    flip_const2 = [z3.BitVec(f"flip_const2_{i}", 64) for i in range(4)]
    flip_const3 = [z3.BitVec(f"flip_const3_{i}", 64) for i in range(4)]

    num_zero = z3.BitVec("num_zero", 64)
    num_one = z3.BitVec("num_one", 64)

    num_threshold = z3.BitVec("num_threshold", 64)

    s.add(
        bb_player[0] == 0x0000001008000000,
        bb_opponent[0] == 0x0000000810000000,
        pop_move[0] == 0,

        directions1[0] == 1,
        directions1[1] == 7,
        directions1[2] == 8,
        directions1[3] == 9,

        directions2[0] == 2,
        directions2[1] == 14,
        directions2[2] == 16,
        directions2[3] == 18,

        flip_const1[0] == 0xFFFFFFFFFFFFFFFF,
        flip_const1[1] == 0x7E7E7E7E7E7E7E7E,
        flip_const1[2] == 0x7E7E7E7E7E7E7E7E,
        flip_const1[3] == 0x7E7E7E7E7E7E7E7E,

        flip_const2[0] == 0x0080808080808080,
        flip_const2[1] == 0x7F00000000000000,
        flip_const2[2] == 0x0102040810204000,
        flip_const2[3] == 0x0040201008040201,

        flip_const3[0] == 0x0101010101010100,
        flip_const3[1] == 0x00000000000000FE,
        flip_const3[2] == 0x0002040810204080,
        flip_const3[3] == 0x8040201008040200,

        num_zero == 0,
        num_one == 1,
        num_threshold == threshold
    )

    for turn in range(MAX_TURN+1):
        bb_occupied = z3.BitVec(f"bb_occupied_{turn}", 64)
        bb_empty = z3.BitVec(f"bb_empty_{turn}", 64)
        masked_bb_opponent = z3.BitVec(f"masked_bb_opponent_{turn}", 64)
    
        movemask = [z3.BitVec(f"movemask_{turn}_{i}", 64) for i in range(4)]

        flip_l_0 = [z3.BitVec(f"flip_l_0_{turn}_{i}", 64) for i in range(4)]
        flip_r_0 = [z3.BitVec(f"flip_r_0_{turn}_{i}", 64) for i in range(4)]
        flip_l_1 = [z3.BitVec(f"flip_l_1_{turn}_{i}", 64) for i in range(4)]
        flip_r_1 = [z3.BitVec(f"flip_r_1_{turn}_{i}", 64) for i in range(4)]
        flip_l_2 = [z3.BitVec(f"flip_l_2_{turn}_{i}", 64) for i in range(4)]
        flip_r_2 = [z3.BitVec(f"flip_r_2_{turn}_{i}", 64) for i in range(4)]
        flip_l_3 = [z3.BitVec(f"flip_l_3_{turn}_{i}", 64) for i in range(4)]
        flip_r_3 = [z3.BitVec(f"flip_r_3_{turn}_{i}", 64) for i in range(4)]
        mask_l = [z3.BitVec(f"mask_l_{turn}_{i}", 64) for i in range(4)]
        mask_r = [z3.BitVec(f"mask_r_{turn}_{i}", 64) for i in range(4)]

        some_moves = [z3.BitVec(f"some_moves_{turn}_{i}", 64) for i in range(4)]
        all_moves = z3.BitVec(f"all_moves_{turn}", 64)
        pop_now = z3.BitVec(f"pop_now_{turn}", 64)

        popcnt_move_tmp1 = z3.BitVec(f"popcnt_move_tmp1_{turn}", 64)
        popcnt_move_tmp2 = z3.BitVec(f"popcnt_move_tmp2_{turn}", 64)

        s.add(
            bb_occupied == bb_player[turn] | bb_opponent[turn],
            bb_empty == bb_occupied ^ 0xFFFFFFFFFFFFFFFF,
            masked_bb_opponent == bb_opponent[turn] & 0x7E7E7E7E7E7E7E7E,
            movemask[0] == masked_bb_opponent,
            movemask[1] == masked_bb_opponent,
            movemask[2] == bb_opponent[turn],
            movemask[3] == masked_bb_opponent
        )
        s.add([z3.And(flip_l_0[i] == (movemask[i] & (bb_player[turn] << directions1[i]))) for i in range(4)])
        s.add([z3.And(flip_r_0[i] == (movemask[i] & z3.LShR(bb_player[turn], directions1[i]))) for i in range(4)])
        s.add([z3.And(flip_l_1[i] == (flip_l_0[i] | (movemask[i] & (flip_l_0[i] << directions1[i])))) for i in range(4)])
        s.add([z3.And(flip_r_1[i] == (flip_r_0[i] | (movemask[i] & z3.LShR(flip_r_0[i], directions1[i])))) for i in range(4)])
        s.add([z3.And(mask_l[i] == (movemask[i] & (movemask[i] << directions1[i]))) for i in range(4)])
        s.add([z3.And(mask_r[i] == (movemask[i] & z3.LShR(movemask[i], directions1[i]))) for i in range(4)])
        s.add([z3.And(flip_l_2[i] == (flip_l_1[i] | (mask_l[i] & (flip_l_1[i] << directions2[i])))) for i in range(4)])
        s.add([z3.And(flip_r_2[i] == (flip_r_1[i] | (mask_r[i] & z3.LShR(flip_r_1[i], directions2[i])))) for i in range(4)])
        s.add([z3.And(flip_l_3[i] == (flip_l_2[i] | (mask_l[i] & (flip_l_2[i] << directions2[i])))) for i in range(4)])
        s.add([z3.And(flip_r_3[i] == (flip_r_2[i] | (mask_r[i] & z3.LShR(flip_r_2[i], directions2[i])))) for i in range(4)])
        s.add([z3.And(some_moves[i] == ((flip_l_3[i] << directions1[i]) | z3.LShR(flip_r_3[i], directions1[i]))) for i in range(4)])
        s.add(
            all_moves == (some_moves[0] | some_moves[1] | some_moves[2] | some_moves[3]) & bb_empty,
            popcnt_move_tmp1 == all_moves - (z3.LShR(all_moves, 1) & 0x7777777777777777) - (z3.LShR(all_moves, 2) & 0x3333333333333333) - (z3.LShR(all_moves, 3) & 0x1111111111111111),
            popcnt_move_tmp2 == ((popcnt_move_tmp1 + z3.LShR(popcnt_move_tmp1, 4)) & 0x0F0F0F0F0F0F0F0F) * 0x0101010101010101,
            pop_now == z3.LShR(popcnt_move_tmp2, 56),
            pop_move[turn + 1] == z3.If(pop_now > pop_move[turn], pop_now, pop_move[turn])
        )

        move_onebit = z3.BitVec(f"move_onebit_{turn}", 64)
        move_bsf_tmp = [z3.BitVec(f"move_bsf_tmp_{turn}_{i}", 64) for i in range(9)]
        move_pos = z3.BitVec(f"move_pos_{turn}", 64)

        s.add(
            move_onebit & (move_onebit - 1) == 0,
            move_onebit & all_moves == move_onebit,

            move_bsf_tmp[0] == move_onebit - 1,
            move_bsf_tmp[1] == move_bsf_tmp[0] | z3.LShR(move_bsf_tmp[0], 1),
            move_bsf_tmp[2] == move_bsf_tmp[1] | z3.LShR(move_bsf_tmp[1], 2),
            move_bsf_tmp[3] == move_bsf_tmp[2] | z3.LShR(move_bsf_tmp[2], 4),
            move_bsf_tmp[4] == move_bsf_tmp[3] | z3.LShR(move_bsf_tmp[3], 8),
            move_bsf_tmp[5] == move_bsf_tmp[4] | z3.LShR(move_bsf_tmp[4], 16),
            move_bsf_tmp[6] == move_bsf_tmp[5] | z3.LShR(move_bsf_tmp[5], 32),
            move_bsf_tmp[7] == move_bsf_tmp[6] - (z3.LShR(move_bsf_tmp[6], 1) & 0x7777777777777777) - (z3.LShR(move_bsf_tmp[6], 2) & 0x3333333333333333) - (z3.LShR(move_bsf_tmp[6], 3) & 0x1111111111111111),
            move_bsf_tmp[8] == ((move_bsf_tmp[7] + z3.LShR(move_bsf_tmp[7], 4)) & 0x0F0F0F0F0F0F0F0F) * 0x0101010101010101,
            move_pos == z3.LShR(move_bsf_tmp[8], 56)
        )

        OM = [z3.BitVec(f"OM_{turn}_{i}", 64) for i in range(4)]
        mask1 = [z3.BitVec(f"mask1_{turn}_{i}", 64) for i in range(4)]
        upperbit_argument = [z3.BitVec(f"upperbit_argument_{turn}_{i}", 64) for i in range(4)]
        upperbit_tmp1 = [z3.BitVec(f"upperbit_tmp1_{turn}_{i}", 64) for i in range(4)]
        upperbit_tmp2 = [z3.BitVec(f"upperbit_tmp2_{turn}_{i}", 64) for i in range(4)]
        upperbit_tmp3 = [z3.BitVec(f"upperbit_tmp3_{turn}_{i}", 64) for i in range(4)]
        upperbit_tmp4 = [z3.BitVec(f"upperbit_tmp4_{turn}_{i}", 64) for i in range(4)]
        upperbit_tmp5 = [z3.BitVec(f"upperbit_tmp5_{turn}_{i}", 64) for i in range(4)]
        upperbit_tmp6 = [z3.BitVec(f"upperbit_tmp6_{turn}_{i}", 64) for i in range(4)]
        upperbit_tmp7 = [z3.BitVec(f"upperbit_tmp7_{turn}_{i}", 64) for i in range(4)]
        upperbit_tmp8 = [z3.BitVec(f"upperbit_tmp8_{turn}_{i}", 64) for i in range(4)]
        upperbit_tmp9 = [z3.BitVec(f"upperbit_tmp9_{turn}_{i}", 64) for i in range(4)]
        upperbit_tmp10 = [z3.BitVec(f"upperbit_tmp10_{turn}_{i}", 64) for i in range(4)]
        upperbit_result = [z3.BitVec(f"upperbit_result_{turn}_{i}", 64) for i in range(4)]
        outflank1 = [z3.BitVec(f"outflank1_{turn}_{i}", 64) for i in range(4)]
        flipped1 = [z3.BitVec(f"flipped1_{turn}_{i}", 64) for i in range(4)]

        mask2 = [z3.BitVec(f"mask2_{turn}_{i}", 64) for i in range(4)]
        outflank2 = [z3.BitVec(f"outflank2_{turn}_{i}", 64) for i in range(4)]
        nonzero = [z3.BitVec(f"nonzero_{turn}_{i}", 64) for i in range(4)]
        flipped2 = [z3.BitVec(f"flipped2_{turn}_{i}", 64) for i in range(4)]

        bb_flip = z3.BitVec(f"bb_flip_{turn}", 64)
        next_player_tmp = z3.BitVec(f"next_player_tmp_{turn}", 64)
        next_opponent_tmp = z3.BitVec(f"next_opponent_tmp_{turn}", 64)

        s.add([z3.And(OM[i] == (bb_opponent[turn] & flip_const1[i])) for i in range(4)])
        s.add([z3.And(mask1[i] == z3.LShR(flip_const2[i], 63 - move_pos)) for i in range(4)])

        s.add([z3.And(upperbit_argument[i] == (~OM[i]) & mask1[i]) for i in range(4)])
        s.add([z3.And(upperbit_tmp1[i] == upperbit_argument[i] | z3.LShR(upperbit_argument[i], 1)) for i in range(4)])
        s.add([z3.And(upperbit_tmp2[i] == upperbit_tmp1[i] | z3.LShR(upperbit_tmp1[i], 2)) for i in range(4)])
        s.add([z3.And(upperbit_tmp3[i] == upperbit_tmp2[i] | z3.LShR(upperbit_tmp2[i], 4)) for i in range(4)])
        s.add([z3.And(upperbit_tmp4[i] == (~z3.LShR(upperbit_tmp3[i], 1)) & upperbit_tmp3[i]) for i in range(4)])
        s.add([z3.And(upperbit_tmp5[i] == (upperbit_tmp4[i] << 32) | z3.LShR(upperbit_tmp4[i], 32)) for i in range(4)])
        s.add([z3.And(upperbit_tmp6[i] == ((upperbit_tmp5[i] & 0x0000FFFF0000FFFF) << 16) | z3.LShR(upperbit_tmp5[i] & 0xFFFF0000FFFF0000, 16)) for i in range(4)])
        s.add([z3.And(upperbit_tmp7[i] == ((upperbit_tmp6[i] & 0x00FF00FF00FF00FF) << 8) | z3.LShR(upperbit_tmp6[i] & 0xFF00FF00FF00FF00, 8)) for i in range(4)])
        s.add([z3.And(upperbit_tmp8[i] == upperbit_tmp7[i] & (-upperbit_tmp7[i])) for i in range(4)])
        s.add([z3.And(upperbit_tmp9[i] == (upperbit_tmp8[i] << 32) | z3.LShR(upperbit_tmp8[i], 32)) for i in range(4)])
        s.add([z3.And(upperbit_tmp10[i] == ((upperbit_tmp9[i] & 0x0000FFFF0000FFFF) << 16) | z3.LShR(upperbit_tmp9[i] & 0xFFFF0000FFFF0000, 16)) for i in range(4)])
        s.add([z3.And(upperbit_result[i] == ((upperbit_tmp10[i] & 0x00FF00FF00FF00FF) << 8) | z3.LShR(upperbit_tmp10[i] & 0xFF00FF00FF00FF00, 8)) for i in range(4)])

        s.add([z3.And(outflank1[i] == upperbit_result[i] & bb_player[turn]) for i in range(4)])
        s.add([z3.And(flipped1[i] == ((-outflank1[i]) << 1) & mask1[i]) for i in range(4)])

        s.add([z3.And(mask2[i] == flip_const3[i] << move_pos) for i in range(4)])
        s.add([z3.And(outflank2[i] == ((OM[i] | (~mask2[i])) + 1) & mask2[i] & bb_player[turn]) for i in range(4)])
        s.add([z3.And(nonzero[i] == z3.If(outflank2[i] == 0, num_zero, num_one)) for i in range(4)])
        s.add([z3.And(flipped2[i] == (outflank2[i] - nonzero[i]) & mask2[i]) for i in range(4)])
        s.add(
            bb_flip == flipped1[0] | flipped1[1] | flipped1[2] | flipped1[3] | flipped2[0] | flipped2[1] | flipped2[2] | flipped2[3],
            next_player_tmp == bb_opponent[turn] ^ bb_flip,
            next_opponent_tmp == bb_player[turn] ^ (bb_flip | move_onebit),
            bb_player[turn + 1] == z3.If(move_onebit == 0, bb_opponent[turn], next_player_tmp),
            bb_opponent[turn + 1] == z3.If(move_onebit == 0, bb_player[turn], next_opponent_tmp)
        )

    s.add(
        pop_move[MAX_TURN+1] >= num_threshold
    )

    t = z3.Then('simplify', 'bit-blast', 'solve-eqs', 'tseitin-cnf')
    subgoal = t(s)
    assert len(subgoal) == 1

    return subgoal[0]

if __name__ == "__main__":

    args = sys.argv
    if len(args) >= 3:
        print("invalid args")
        sys.exit(1)
    threshold = 34
    if len(args) == 2:
        threshold = int(args[1])

    print("threshold = " + str(threshold))
    print("start: solve")
    subgoal = solve(threshold)
    print("finish: solve")
    print("start: write a raw cnf (it may take tens of minutes...)")

    with open("cnf_subgoals_" + str(threshold) + ".txt", "w", encoding='utf-8') as f:

        for x in subgoal:
            s = "".join(str(x).split())
            f.write(s + "\n")

    print("finish: write a raw cnf")

    k_name_to_number = dict()
    clauses = []

    print("start: convert it into DIMACS CNF format")

    with open("cnf_subgoals_" + str(threshold) + ".txt", "r", encoding='utf-8') as f:

        line = f.readline()
        while line:
            answer = line.strip()
            answer = answer.replace(",","),")
            
            x = re.findall(r"k![0-9]+\)", answer)
            for s in x:
                if s not in k_name_to_number:
                    k_name_to_number[s] = str(len(k_name_to_number) + 1)
                answer = answer.replace(s, k_name_to_number[s])
            answer = answer.replace("Or(","")
            answer = answer.replace("Not","-")
            answer = answer.replace("(","")
            answer = answer.replace(")","")
            answer = answer.replace(","," ")
            clauses.append(answer)

            line = f.readline()

    with open("dimacs_cnf_" + str(threshold) + ".txt", "w", encoding='utf-8') as f:
        f.write("p cnf " + str(len(k_name_to_number)) + " " + str(len(clauses)) + "\n")
        for x in clauses:
            f.write(x + " 0\n")

    print("finished")

# 15849027 clauses
# 570969 literals