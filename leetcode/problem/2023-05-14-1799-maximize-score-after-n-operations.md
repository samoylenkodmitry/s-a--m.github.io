---
layout: leetcode-entry
title: "1799. Maximize Score After N Operations"
permalink: "/leetcode/problem/2023-05-14-1799-maximize-score-after-n-operations/"
leetcode_ui: true
entry_slug: "2023-05-14-1799-maximize-score-after-n-operations"
---

[1799. Maximize Score After N Operations](https://leetcode.com/problems/maximize-score-after-n-operations/description/) hard
[blog post](https://leetcode.com/problems/maximize-score-after-n-operations/solutions/3522041/kotiln-dfs-cache-bitmask-gcd/)
[substack](https://dmitriisamoilenko.substack.com/p/14052023-1799-maximize-score-after?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/212
#### Problem TLDR
Max indexed-gcd-pair sum from 2n array; [3,4,6,8] -> 11 (1gcd(3,6) + 2gcd(4,8))
#### Intuition
For each `step` and remaining items, the result is always the same, so is memorizable.

#### Approach
* search all possible combinations with DFS
* use `bitmask` to avoid double counting
* use an array for cache
#### Complexity
- Time complexity:
$$O(n^22^n)$$
- Space complexity:
$$O(n2^n)$$

#### Code

```kotlin

    fun gcd(a: Int, b: Int): Int = if (b % a == 0) a else gcd(b % a, a)
    fun maxScore(nums: IntArray): Int {
        val n = nums.size / 2
        val cache = Array(n + 1) { IntArray(1 shl nums.size) { -1 } }
        fun dfs(step: Int, mask: Int): Int {
            if (step > n) return 0
            if (cache[step][mask] != -1) return cache[step][mask]
            var max = 0
            for (i in 0..nums.lastIndex) {
                val ibit = 1 shl i
                if (mask and ibit != 0) continue
                for (j in (i + 1)..nums.lastIndex) {
                    val jbit = 1 shl j
                    if (mask and jbit != 0) continue
                    val curr = step * gcd(nums[i], nums[j])
                    val next = dfs(step + 1, mask or ibit or jbit)
                    max = maxOf(max, curr + next)
                }
            }
            cache[step][mask] = max
            return max
        }
        return dfs(1, 0)
    }

```

