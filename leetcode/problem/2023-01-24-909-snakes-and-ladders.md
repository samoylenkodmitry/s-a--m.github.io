---
layout: leetcode-entry
title: "909. Snakes and Ladders"
permalink: "/leetcode/problem/2023-01-24-909-snakes-and-ladders/"
leetcode_ui: true
entry_slug: "2023-01-24-909-snakes-and-ladders"
---

[909. Snakes and Ladders](https://leetcode.com/problems/snakes-and-ladders/description/) medium

[https://t.me/leetcode_daily_unstoppable/96](https://t.me/leetcode_daily_unstoppable/96)

[blog post](https://leetcode.com/problems/snakes-and-ladders/solutions/3094842/kotlin-bfs/)

```kotlin
    fun snakesAndLadders(board: Array<IntArray>): Int {
        fun col(pos: Int): Int {
            return if (((pos/board.size) % 2) == 0)
                    (pos % board.size)
                else
                    (board.lastIndex - (pos % board.size))
        }
        val last = board.size * board.size
        var steps = 0
        val visited = mutableSetOf<Int>()
        with(ArrayDeque<Int>().apply { add(1) }) {
            while (isNotEmpty() && steps <= last) {
                repeat(size) {
                    var curr = poll()
                    val jump = board[board.lastIndex - (curr-1)/board.size][col(curr-1)]
                    if (jump != -1) curr = jump
                    if (curr == last) return steps
                    for (i in 1..6)
                        if (visited.add(curr + i) && curr + i <= last) add(curr + i)
                }
                steps++
            }
        }
        return -1
    }

```

In each step, we can choose the best outcome, so we need to travel all of them in the parallel and calculate steps number. This is a BFS.

We can avoid that strange order by iterating it and store into the linear array. Or just invent a formula for row and column by given index.

Space: O(n^2), Time: O(n^2), n is a grid size

