---
layout: leetcode-entry
title: "1444. Number of Ways of Cutting a Pizza"
permalink: "/leetcode/problem/2023-03-31-1444-number-of-ways-of-cutting-a-pizza/"
leetcode_ui: true
entry_slug: "2023-03-31-1444-number-of-ways-of-cutting-a-pizza"
---

[1444. Number of Ways of Cutting a Pizza](https://leetcode.com/problems/number-of-ways-of-cutting-a-pizza/description/) hard

[blog post](https://leetcode.com/problems/number-of-ways-of-cutting-a-pizza/solutions/3361348/kotlin-dfs-memo-prefixsum/)

```kotlin

data class Key(val x: Int, val y: Int, val c: Int)
fun ways(pizza: Array<String>, k: Int): Int {
    val havePizza = Array(pizza.size) { Array<Int>(pizza[0].length) { 0 } }

        val lastX = pizza[0].lastIndex
        val lastY = pizza.lastIndex
        for (y in lastY downTo 0) {
            var sumX = 0
            for (x in lastX downTo 0) {
                sumX += if (pizza[y][x] == 'A') 1 else 0
                havePizza[y][x] = sumX + (if (y == lastY) 0 else havePizza[y + 1][x])
            }
        }

        val cache = mutableMapOf<Key, Int>()
        fun dfs(x: Int, y: Int, c: Int): Int {
            return cache.getOrPut(Key(x, y, c)) {
                if (c == 0) return@getOrPut if (havePizza[y][x] > 0) 1 else 0
                var res = 0
                for (xx in x + 1..lastX) if (havePizza[y][x] > havePizza[y][xx])
                res = (res + dfs(xx, y, c - 1)) % 1_000_000_007

                for (yy in y + 1..lastY) if (havePizza[y][x] > havePizza[yy][x])
                res = (res + dfs(x, yy, c - 1)) % 1_000_000_007

                return@getOrPut res
            }
        }
        return dfs(0, 0, k - 1)
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/165
#### Intuition
The tricky problem is to how to program a number of cuts.
We can do the horizontal and vertical cuts decreasing available number `k` and tracking if we have any apples `before` and any apples `after` the cut. To track this, we can precompute a `prefix sum` of the apples, by each `top-left` corner to the end of the pizza. The stopping condition of the DFS is if we used all available cuts.

#### Approach
* carefully precompute prefix sum. You move by row, increasing `sumX`, then you move by column and reuse the result of the previous row.
* to detect if there are any apples above or to the left, compare the total number of apples precomputed from the start of the given `x,y` in the arguments and from the other side of the cut `xx,y` or `x, yy`.
#### Complexity
- Time complexity:
$$O(mnk(m+n))$$, mnk - number of cached states, (m+n) - search in each DFS step
- Space complexity:
$$O(mnk)$$

