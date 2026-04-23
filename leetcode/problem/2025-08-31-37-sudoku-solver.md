---
layout: leetcode-entry
title: "37. Sudoku Solver"
permalink: "/leetcode/problem/2025-08-31-37-sudoku-solver/"
leetcode_ui: true
entry_slug: "2025-08-31-37-sudoku-solver"
---

[37. Sudoku Solver](https://leetcode.com/problems/sudoku-solver/description/) hard
[blog post](https://leetcode.com/problems/sudoku-solver/solutions/7141177/kotlin-by-samoylenkodmitry-feha/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/31082025-37-sudoku-solver?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/7nrj41U5HTg)

![1.webp](/assets/leetcode_daily_images/8e1d3e7e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1098

#### Problem TLDR

Solve sudoku #hard #backtrack

#### Intuition

Brute-force DFS with pruning

#### Approach

* we don't have to validate in the end; choose only available numbers
* there is a bitmask optimization
* we can prioritize rows, cols or subs with more numbers filled

#### Complexity

- Time complexity:
$$O(9^81)$$, however, `9` is smaller with pruning

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 233ms
    fun solveSudoku(b: Array<CharArray>): Unit {
        val rows = Array(9) { b[it].toHashSet() }
        val cols = Array(9) { x -> (0..8).map { b[it][x] }.toHashSet() }
        val subs = Array(9) { i -> (0..8).map { j ->  b[i/3*3 + j/3][i%3*3 + j%3] }.toHashSet() }
        fun dfs(i: Int): Boolean {
            if (i == 81) return true
            val y = i / 9; val x = i % 9
            if (b[y][x] != '.') return dfs(i + 1)
            for (c in '1'..'9') if (c !in rows[y] && c !in cols[x] && c !in subs[y/3*3+x/3]) {
                b[y][x] = c; rows[y] += c; cols[x] += c; subs[y/3*3+x/3] +=c
                if (dfs(i + 1)) return true
                rows[y] -= c; cols[x] -= c; subs[y/3*3+x/3] -=c
            }
            b[y][x] = '.'; return false
        }
        dfs(0)
    }

```
```kotlin

// 79ms
    fun solveSudoku(b: Array<CharArray>): Unit {
        val s = Array(3) { IntArray(9) }
        for (y in 0..8) for (x in 0..8) if (b[y][x] != '.') {
            val c = 1 shl (b[y][x] - '0')
            s[0][y] = s[0][y] or c; s[1][x] = s[1][x] or c; s[2][y/3*3+x/3] = s[2][y/3*3+x/3] or c}
        fun dfs(i: Int): Boolean {
            if (i == 81) return true
            val y = i / 9; val x = i % 9
            if (b[y][x] != '.') return dfs(i + 1)
            for (n in 1..9) {
                val c = 1 shl n
                if ((c and s[0][y]) + (c and s[1][x]) + (c and s[2][y/3*3+x/3]) == 0) {
                s[0][y] = s[0][y] xor c; s[1][x] = s[1][x] xor c; s[2][y/3*3+x/3] = s[2][y/3*3+x/3] xor c
                b[y][x] = '0' + n; if (dfs(i + 1)) return true
                s[0][y] = s[0][y] xor c; s[1][x] = s[1][x] xor c; s[2][y/3*3+x/3] = s[2][y/3*3+x/3] xor c
            }}
            b[y][x] = '.'; return false
        }
        dfs(0)
    }

```

