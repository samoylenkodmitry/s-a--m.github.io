---
layout: leetcode-entry
title: "377. Combination Sum IV"
permalink: "/leetcode/problem/2023-09-09-377-combination-sum-iv/"
leetcode_ui: true
entry_slug: "2023-09-09-377-combination-sum-iv"
---

[377. Combination Sum IV](https://leetcode.com/problems/combination-sum-iv/description/) medium
[blog post](https://leetcode.com/problems/combination-sum-iv/solutions/4020533/kotlin-dfs-cache/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/9092023-377-combination-sum-iv?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/368432b4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/334

#### Problem TLDR

Number of ways to sum up array nums to target

#### Intuition

This is a canonical DP knapsack problem: choose one of the items and decrease the `target` by its value. If `target` is zero - we have a single way, if negative - no ways, otherwise keep taking items. The result will only depend on the `target`, so can be cached.

#### Approach

In this code:
* trick to make conversion `0 -> 1, negative -> 0`: `1 - (t ushr 31)`, it shifts the leftmost bit to the right treating sign bit as a value bit, converting any negative number to `1` and positive to `0`
* `IntArray` used instead of `Map` using `takeIf` Kotlin operator

#### Complexity

- Time complexity:
$$O(n^2)$$, `n` for the recursion depth, and `n` for the inner iteration

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

    fun combinationSum4(nums: IntArray, target: Int): Int {
      val cache = IntArray(target + 1) { -1 }
      fun dfs(t: Int): Int = if (t <= 0) 1 - (t ushr 31) else
        cache[t].takeIf { it >= 0 } ?:
        nums.sumBy { dfs(t - it) }.also { cache[t] = it }
      return dfs(target)
    }

```

