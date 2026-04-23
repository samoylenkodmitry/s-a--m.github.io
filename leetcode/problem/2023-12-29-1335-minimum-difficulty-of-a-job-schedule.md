---
layout: leetcode-entry
title: "1335. Minimum Difficulty of a Job Schedule"
permalink: "/leetcode/problem/2023-12-29-1335-minimum-difficulty-of-a-job-schedule/"
leetcode_ui: true
entry_slug: "2023-12-29-1335-minimum-difficulty-of-a-job-schedule"
---

[1335. Minimum Difficulty of a Job Schedule](https://leetcode.com/problems/minimum-difficulty-of-a-job-schedule/description/) hard
[blog post](https://leetcode.com/problems/minimum-difficulty-of-a-job-schedule/solutions/4473265/kotlin-dp/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29122023-1335-minimum-difficulty?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/WvIIpPh9UZo)
![image.png](/assets/leetcode_daily_images/dfafa235.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/454

#### Problem TLDR

Min sum of maximums jobDifficulty[i] per day d preserving the order

#### Intuition

Let's brute-force optimal interval of jobs `jobInd..j` for every day using Depth-First Search. The result will only depend on the starting `jobInd` and the current `day`, so can be cached.

#### Approach

* pay attention to the problem description, preserving jobs order matters here

#### Complexity

- Time complexity:
$$O(dn^2)$$, `dn` for the recursion depth and another `n` for the inner loop

- Space complexity:
$$O(dn)$$

#### Code

```kotlin

  fun minDifficulty(jobDifficulty: IntArray, d: Int): Int {
    val dp = mutableMapOf<Pair<Int, Int>, Int>()
    fun dfs(jobInd: Int, day: Int): Int = when {
      jobInd == jobDifficulty.size -> if (day == d) 0 else Int.MAX_VALUE / 2
      day == d -> Int.MAX_VALUE / 2
      else -> dp.getOrPut(jobInd to day) {
        var max = 0
        (jobInd..jobDifficulty.lastIndex).minOf { i ->
          max = max(max, jobDifficulty[i])
          max + dfs(i + 1, day + 1)
        }
    }}
    return dfs(0, 0).takeIf { it < Int.MAX_VALUE / 2 } ?: -1
  }

```

