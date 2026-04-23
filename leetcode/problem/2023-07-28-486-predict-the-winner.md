---
layout: leetcode-entry
title: "486. Predict the Winner"
permalink: "/leetcode/problem/2023-07-28-486-predict-the-winner/"
leetcode_ui: true
entry_slug: "2023-07-28-486-predict-the-winner"
---

[486. Predict the Winner](https://leetcode.com/problems/predict-the-winner/description/) medium
[blog post](https://leetcode.com/problems/predict-the-winner/solutions/3826663/kotlin-dfs-cache/)
[substack](https://dmitriisamoilenko.substack.com/p/28072023-486-predict-the-winner?sd=pf)
![image.png](/assets/leetcode_daily_images/5feb00cc.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/289

#### Problem TLDR

Optimally taking numbers from an `array's ends` can one player win another

#### Intuition

The optimal strategy for the current player will be to search the maximum score of `total sum - optimal another`. The result can be cached as it only depends on the input array.

#### Approach

Write the DFS and cache by `lo` and `hi`.
* use `Long` to avoid overflow

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

    fun PredictTheWinner(nums: IntArray): Boolean {
      val cache = Array(nums.size) { LongArray(nums.size) { -1L } }
      fun dfs(lo: Int, hi: Int, currSum: Long): Long = cache[lo][hi].takeIf { it >= 0 } ?: {
        if (lo == hi) nums[lo].toLong()
        else if (lo > hi) 0L
        else currSum - minOf(
          dfs(lo + 1, hi, currSum - nums[lo]),
          dfs(lo, hi - 1, currSum - nums[hi])
        )
      }().also { cache[lo][hi] = it }
      val sum = nums.asSequence().map { it.toLong() }.sum()!!
      return dfs(0, nums.lastIndex, sum).let { it >= sum - it }
    }

```

