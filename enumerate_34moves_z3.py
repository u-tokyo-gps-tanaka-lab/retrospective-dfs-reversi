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

def print_board(player, opponent, move):
    print("  A B C D E F G H")
    for i in range(8):
        print(str(i+1) + " ", end = "")
        for j in range(8):
            if (player & (1 << (i * 8 + j))) > 0:
                print("* ", end = "")
            elif (opponent & (1 << (i * 8 + j))) > 0:
                print("o ", end = "")
            elif (move & (1 << (i * 8 + j))) > 0:
                print(". ", end = "")
            else:
                print("- ", end = "")
        print(i+1, end = "")
        
        if i == 2:
            print("  * = player's disc", end = "")
        elif i == 3:
            print("  o = opponent's disc", end = "")
        elif i == 4:
            print("  - = empty, illegal move", end = "")
        elif i == 5:
            print("  . = empty, legal move", end = "")
        
        print("")
    print("  A B C D E F G H")

def solve(movenum, known_answer_list):

    bb_player = z3.BitVec("bb_player", 64)
    bb_opponent = z3.BitVec("bb_opponent", 64)
    bb_occupied = z3.BitVec("bb_occupied", 64)
    bb_empty = z3.BitVec("bb_empty", 64)
    masked_bb_opponent = z3.BitVec("masked_bb_opponent", 64)
    
    movemask = [z3.BitVec(f"movemask_{i}", 64) for i in range(4)]

    directions1 = [z3.BitVec(f"directions1_{i}", 64) for i in range(4)]
    directions2 = [z3.BitVec(f"directions2_{i}", 64) for i in range(4)]

    flip_l_0 = [z3.BitVec(f"flip_l_0_{i}", 64) for i in range(4)]
    flip_r_0 = [z3.BitVec(f"flip_r_0_{i}", 64) for i in range(4)]
    flip_l_1 = [z3.BitVec(f"flip_l_1_{i}", 64) for i in range(4)]
    flip_r_1 = [z3.BitVec(f"flip_r_1_{i}", 64) for i in range(4)]
    flip_l_2 = [z3.BitVec(f"flip_l_2_{i}", 64) for i in range(4)]
    flip_r_2 = [z3.BitVec(f"flip_r_2_{i}", 64) for i in range(4)]
    flip_l_3 = [z3.BitVec(f"flip_l_3_{i}", 64) for i in range(4)]
    flip_r_3 = [z3.BitVec(f"flip_r_3_{i}", 64) for i in range(4)]
    mask_l = [z3.BitVec(f"mask_l_{i}", 64) for i in range(4)]
    mask_r = [z3.BitVec(f"mask_r_{i}", 64) for i in range(4)]

    some_moves = [z3.BitVec(f"some_moves_{i}", 64) for i in range(4)]
    all_moves = z3.BitVec("all_moves", 64)

    popcnt_move_tmp1 = z3.BitVec("popcnt_move_tmp1", 64)
    popcnt_move_tmp2 = z3.BitVec("popcnt_move_tmp2", 64)
    pop_move = z3.BitVec("pop_move", 64)

    popcnt_empty_tmp1 = z3.BitVec("popcnt_empty_tmp1", 64)
    popcnt_empty_tmp2 = z3.BitVec("popcnt_empty_tmp2", 64)
    pop_empty = z3.BitVec("pop_empty", 64)


    s = z3.Solver()
    s.add(
        (bb_player & bb_opponent) == 0,
        bb_occupied == bb_player | bb_opponent,
        bb_empty == bb_occupied ^ 0xFFFFFFFFFFFFFFFF,
        (bb_occupied & 0x0000001818000000) == 0x0000001818000000,
        masked_bb_opponent == bb_opponent & 0x7E7E7E7E7E7E7E7E,
        movemask[0] == masked_bb_opponent,
        movemask[1] == masked_bb_opponent,
        movemask[2] == bb_opponent,
        movemask[3] == masked_bb_opponent,
        directions1[0] == 1,
        directions1[1] == 7,
        directions1[2] == 8,
        directions1[3] == 9,
        directions2[0] == 2,
        directions2[1] == 14,
        directions2[2] == 16,
        directions2[3] == 18
    )
    s.add([z3.And(flip_l_0[i] == (movemask[i] & (bb_player << directions1[i]))) for i in range(4)])
    s.add([z3.And(flip_r_0[i] == (movemask[i] & z3.LShR(bb_player, directions1[i]))) for i in range(4)])
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
        pop_move == z3.LShR(popcnt_move_tmp2, 56),
        pop_move >= movenum
    )

    s.add(
        # !!! heuristic !!!
        (bb_empty & 0xFF818181818181FF) == 0xFF818181818181FF
    )

    if len(known_answer_list) > 0:
        known_answer_p = [z3.BitVec(f"known_answer_p_{i}", 64) for i in range(len(known_answer_list))]
        known_answer_o = [z3.BitVec(f"known_answer_o_{i}", 64) for i in range(len(known_answer_list))]
        s.add([z3.And(known_answer_p[i] == known_answer_list[i][0]) for i in range(len(known_answer_list))])
        s.add([z3.And(known_answer_o[i] == known_answer_list[i][1]) for i in range(len(known_answer_list))])
        s.add([z3.And(((known_answer_p[i] ^ bb_player) | (known_answer_o[i] ^ bb_opponent)) != 0) for i in range(len(known_answer_list))])

    time_start = time.time()
    result = s.check()
    time_end = time.time()

    print(f"board: {movenum} or more moves: {result}. elapsed time = {int(time_end - time_start)} second")

    if result == z3.unsat: return ()

    satlist = sorted([x.strip() for x in str(s.model())[1:-1].split(",")]) # ['all_moves = 18446744073709551615', 'bb_empty = 8', ...
    satlist = sorted([x.split(" = ") for x in satlist if re.match(r"^(all_moves|bb_player|bb_opponent) = ", x.strip()) is not None])
    for x in satlist:
        print(x[0] + "".join([" " for _ in range(11 - len(x[0]))]) + " = " + format(int(x[1]), "#066b"))

    u64_player = int([x for x in satlist if "player" in x[0]][0][1])
    u64_opponent = int([x for x in satlist if "opponent" in x[0]][0][1])
    u64_moves = int([x for x in satlist if "moves" in x[0]][0][1])
    print_board(u64_player,u64_opponent,u64_moves)

    return (u64_player, u64_opponent)


if __name__ == "__main__":

    known_answer = []

    data = pd.read_csv("reversi34moves.csv", na_filter=False, encoding="utf-8")
    for index, row in data.iterrows():
        p = row["player"]
        o = row["opponent"]
        known_answer.append((int(p,16),int(o,16)))

    for i in range(200):
        x = solve(34, known_answer)
        if len(x) != 2: break
        known_answer.append(x)

    with open("reversi34moves.csv", "w", encoding='utf-8') as f:
        f.write("player,opponent\n")
        for x in known_answer:
            f.write(format(x[0],"#018x") + "," + format(x[1],"#018x") + "\n")

    print("finished")
