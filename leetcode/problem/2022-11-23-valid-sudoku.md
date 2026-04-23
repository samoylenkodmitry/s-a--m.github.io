---
layout: leetcode-entry
title: "Valid Sudoku"
permalink: "/leetcode/problem/2022-11-23-valid-sudoku/"
leetcode_ui: true
entry_slug: "2022-11-23-valid-sudoku"
---

[https://leetcode.com/problems/valid-sudoku/](https://leetcode.com/problems/valid-sudoku/) medium

```kotlin

    fun isValidSudoku(board: Array<CharArray>): Boolean {
        val cell9 = arrayOf(0 to 0, 0 to 1, 0 to 2,
                            1 to 0, 1 to 1, 1 to 2,
                            2 to 0, 2 to 1, 2 to 2)
        val starts = arrayOf(0 to 0, 0 to 3, 0 to 6,
                             3 to 0, 3 to 3, 3 to 6,
                             6 to 0, 6 to 3, 6 to 6)
        return !starts.any { (sy, sx) ->
                val visited = HashSet<Char>()
                cell9.any { (dy, dx) ->
                    val c = board[sy+dy][sx+dx]
                    c != '.' && !visited.add(c)
                }
            } && !board.any { row ->
                val visited = HashSet<Char>()
                row.any { it != '.' && !visited.add(it) }
            } && !(0..8).any { x ->
                val visited = HashSet<Char>()
                (0..8).any { board[it][x] != '.' && !visited.add(board[it][x]) }
            }
    }

```

This is an easy problem, just do what is asked.

Complexity: O(N)
Memory: O(N), N = 81, so it O(1)

