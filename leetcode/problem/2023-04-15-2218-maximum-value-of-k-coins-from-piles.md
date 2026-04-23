---
layout: leetcode-entry
title: "2218. Maximum Value of K Coins From Piles"
permalink: "/leetcode/problem/2023-04-15-2218-maximum-value-of-k-coins-from-piles/"
leetcode_ui: true
entry_slug: "2023-04-15-2218-maximum-value-of-k-coins-from-piles"
---

[2218. Maximum Value of K Coins From Piles](https://leetcode.com/problems/maximum-value-of-k-coins-from-piles/description/) hard

```kotlin

fun maxValueOfCoins(piles: List<List<Int>>, k: Int): Int {
    val cache = Array(piles.size) { mutableListOf<Long>() }

        fun dfs(pile: Int, taken: Int): Long {
            if (taken >= k || pile >= piles.size) return 0
            if (cache[pile].size > taken) return cache[pile][taken]
            var max = dfs(pile + 1, taken)
            var sum = 0L
            for (j in 0..piles[pile].lastIndex) {
                val newTaken = taken + j + 1
                if (newTaken > k) break
                sum += piles[pile][j]
                max = maxOf(max, sum + dfs(pile + 1, newTaken))
            }
            cache[pile].add(max)
            return max
        }

        return dfs(0, 0).toInt()
    }

```

[blog post](https://leetcode.com/problems/maximum-value-of-k-coins-from-piles/solutions/3418459/kotlin-dfs-cache/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-15042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/181
#### Intuition
Given the current pile, we can assume there is an optimal maximum value of the piles to the right of the current for every given number of `k`.
![leetcode_daily_backtrack.gif](/assets/leetcode_daily_images/6e541f99.webp)

#### Approach
We can cache the result by the keys of every `pile to taken`

#### Complexity
- Time complexity:
$$O(kn^2)$$
- Space complexity:
$$O(kn^2)$$

