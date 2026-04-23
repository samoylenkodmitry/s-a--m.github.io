---
layout: leetcode-entry
title: "446. Arithmetic Slices II - Subsequence"
permalink: "/leetcode/problem/2024-01-07-446-arithmetic-slices-ii-subsequence/"
leetcode_ui: true
entry_slug: "2024-01-07-446-arithmetic-slices-ii-subsequence"
---

[446. Arithmetic Slices II - Subsequence](https://leetcode.com/problems/arithmetic-slices-ii-subsequence/description/) hard
[blog post](https://leetcode.com/problems/arithmetic-slices-ii-subsequence/solutions/4521808/kotlin-dp/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/7012024-446-arithmetic-slices-ii?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/3kFB0lC8oxM)
![image.png](/assets/leetcode_daily_images/94c7275a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/464

#### Problem TLDR

Count of arithmetic subsequences.

#### Intuition

We can take every pair and search for the third element.
The result only depends on the `diff` and suffix array position, so can be cached.

#### Approach

* be careful how to count each new element: first add the `1` then add the suffix count. Wrong approach: just count the `1` at the end of the sequence.

#### Complexity

- Time complexity:
$$O(n^2)$$, it looks like n^4, but the `dfs` n^2 part will only go deep once

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

  fun numberOfArithmeticSlices(nums: IntArray): Int {
    val dp = mutableMapOf<Pair<Int, Int>, Int>()
    fun dfs(i: Int, k: Int): Int = dp.getOrPut(i to k) {
      var count = 0
      for (j in i + 1..<nums.size)
        if (nums[i].toLong() - nums[k] == nums[j].toLong() - nums[i])
          count += 1 + dfs(j, i)
      count
    }
    var count = 0
    for (i in nums.indices)
      for (j in i + 1..<nums.size)
        count += dfs(j, i)
    return count
  }

```

