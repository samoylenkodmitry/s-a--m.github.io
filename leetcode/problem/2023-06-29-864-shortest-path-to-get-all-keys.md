---
layout: leetcode-entry
title: "864. Shortest Path to Get All Keys"
permalink: "/leetcode/problem/2023-06-29-864-shortest-path-to-get-all-keys/"
leetcode_ui: true
entry_slug: "2023-06-29-864-shortest-path-to-get-all-keys"
---

[864. Shortest Path to Get All Keys](https://leetcode.com/problems/shortest-path-to-get-all-keys/description/) hard
[blog post](https://leetcode.com/problems/shortest-path-to-get-all-keys/solutions/3695847/kotlin-bfs/)
[substack](https://dmitriisamoilenko.substack.com/p/29062023-864-shortest-path-to-get?sd=pf)
![image.png](/assets/leetcode_daily_images/05c4c9d8.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/260
#### Problem TLDR
Min steps to collect all `lowercase` keys in matrix. `#` and `uppercase` locks are blockers.
#### Intuition
What will not work:
* dynamic programming – gives TLE
* DFS – as we can visit cells several times

For the shortest path, we can make a Breadth-First Search wave in a space of the current position and collected keys set.

#### Approach
* Let's use bit mask for collected keys set
* all bits set are `(1 << countKeys) - 1`

#### Complexity

- Time complexity:
$$O(nm2^k)$$

- Space complexity:
$$O(nm2^k)$$

#### Code

```kotlin

val dir = arrayOf(0, 1, 0, -1)
data class Step(val y: Int, val x: Int, val keys: Int)
fun shortestPathAllKeys(grid: Array<String>): Int {
    val w = grid[0].length
    val s = (0..grid.size * w).first { '@' == grid[it / w][it % w] }
    val bit: (Char) -> Int = { 1 shl (it.toLowerCase().toInt() - 'a'.toInt()) }
    val visited = HashSet<Step>()
        val allKeys = (1 shl (grid.map { it.count { it.isLowerCase() } }.sum()!!)) - 1
        var steps = -1
        return with(ArrayDeque<Step>()) {
            add(Step(s / w, s % w, 0))
            while (isNotEmpty() && steps++ < grid.size * w) {
                repeat(size) {
                    val step = poll()
                    val (y, x, keys) = step
                    if (keys == allKeys) return steps - 1
                    if (x in 0 until w && y in 0..grid.lastIndex && visited.add(step)) {
                        val cell = grid[y][x]
                        if (cell != '#' && !(cell.isUpperCase() && 0 == (keys and bit(cell)))) {
                            val newKeys = if (cell.isLowerCase()) (keys or bit(cell)) else keys
                            var dx = -1
                            dir.forEach { dy ->  add(Step(y + dy, x + dx, newKeys)).also { dx = dy } }
                        }
                    }
                }
            }
            -1
        }
    }

```

