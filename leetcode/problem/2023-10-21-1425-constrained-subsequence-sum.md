---
layout: leetcode-entry
title: "1425. Constrained Subsequence Sum"
permalink: "/leetcode/problem/2023-10-21-1425-constrained-subsequence-sum/"
leetcode_ui: true
entry_slug: "2023-10-21-1425-constrained-subsequence-sum"
---

[1425. Constrained Subsequence Sum](https://leetcode.com/problems/constrained-subsequence-sum/description/) hard
[blog post](https://leetcode.com/problems/constrained-subsequence-sum/solutions/4191510/kotlin-decreasing-queue-evolve/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22102023-1425-constrained-subsequence?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/d3aa6326.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/377

#### Problem TLDR

Max sum of subsequence `i - j <= k`

#### Intuition

The naive DP approach is to do DFS and memoization:

```kotlin
    fun constrainedSubsetSum(nums: IntArray, k: Int): Int {
      val dp = mutableMapOf<Int, Int>()
      fun dfs(i: Int): Int = if (i >= nums.size) 0 else dp.getOrPut(i) {
        var max = nums[i]
        for (j in 1..k) max = max(max, nums[i] + dfs(i + j))
        max
      }
      return (0..<nums.size).maxOf { dfs(it) }
    }
```

This solution takes O(nk) time and gives TLE.

Let's rewrite it to the iterative version to think about further optimization:

```kotlin
    fun constrainedSubsetSum(nums: IntArray, k: Int): Int {
      val dp = mutableMapOf<Int, Int>()
      for (i in nums.indices)
        dp[i] = nums[i] + (i - k..i).maxOf { dp[it] ?: 0 }
      return dp.values.max()
    }
```

Next, read a hint :) It will suggest to use a Heap. Indeed, looking at this code, we're just choosing a maximum value from the last `k` values:

```kotlin
    fun constrainedSubsetSum(nums: IntArray, k: Int): Int =
    with (PriorityQueue<Int>(reverseOrder())) {
      val dp = mutableMapOf<Int, Int>()
      for (i in nums.indices) {
        if (i - k > 0) remove(dp[i - k - 1])
        dp[i] = nums[i] + max(0, peek() ?: 0)
        add(dp[i])
      }
      dp.values.max()
    }
```

This solution takes O(nlog(k)) time and still gives TLE.

Let's look at other's people solutions, just take a hint: `decreasing queue`. This technique must be remembered, as it is a common trick to find a maximum in a sliding window with O(1) time.

#### Approach

`Decreasing queue` flushes all the values that smaller than the current.
* we'll store the indices to remove them later if out of `k`
* careful with indices

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(k)$$

#### Code

```kotlin

    fun constrainedSubsetSum(nums: IntArray, k: Int): Int =
    with (ArrayDeque<Int>()) {
      for (i in nums.indices) {
        if (isNotEmpty() && first() < i - k) removeFirst()
        if (isNotEmpty()) nums[i] += max(0, nums[first()])
        while (isNotEmpty() && nums[last()] < nums[i]) removeLast()
        addLast(i)
      }
      nums.max()
    }

```

