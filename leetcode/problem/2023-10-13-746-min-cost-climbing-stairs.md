---
layout: leetcode-entry
title: "746. Min Cost Climbing Stairs"
permalink: "/leetcode/problem/2023-10-13-746-min-cost-climbing-stairs/"
leetcode_ui: true
entry_slug: "2023-10-13-746-min-cost-climbing-stairs"
---

[746. Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs/description/) easy
[blog post](https://leetcode.com/problems/min-cost-climbing-stairs/solutions/4163218/kotlin-dp/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13102023-746-min-cost-climbing-stairs?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/980347fd.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/368

#### Problem TLDR

Classic DP: climbing stairs

#### Intuition

Start with brute force approach: consider every position and choose one of a two - use current stair or use next. Given that, the result will only depend on the input position, so can be cached. This will give a simple DFS + memo DP code:
```kotlin
    fun minCostClimbingStairs(cost: IntArray): Int {
      val dp = mutableMapOf<Int, Int>()
      fun dfs(curr: Int): Int = dp.getOrPut(curr) {
        if (curr >= cost.lastIndex) 0
        else min(
          cost[curr] + dfs(curr + 1),
          cost[curr + 1] + dfs(curr + 2)
        )
      }
      return dfs(0)
    }
```
This is accepted, but can be better if rewritten to bottom up and optimized.

#### Approach

After rewriting the recursive solution to iterative bottom up, we can notice, that only `two` of the previous values are always used. Convert dp array into two variables.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minCostClimbingStairs(cost: IntArray): Int {
      var curr = 0
      var prev = 0
      for (i in 0..<cost.lastIndex)
        curr = min(cost[i + 1] + curr, cost[i] + prev)
              .also { prev = curr }
      return curr
    }

```

