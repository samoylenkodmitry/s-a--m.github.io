---
layout: leetcode-entry
title: "1235. Maximum Profit in Job Scheduling"
permalink: "/leetcode/problem/2024-01-06-1235-maximum-profit-in-job-scheduling/"
leetcode_ui: true
entry_slug: "2024-01-06-1235-maximum-profit-in-job-scheduling"
---

[1235. Maximum Profit in Job Scheduling](https://leetcode.com/problems/maximum-profit-in-job-scheduling/description/) hard
[blog post](https://leetcode.com/problems/maximum-profit-in-job-scheduling/solutions/4516146/kotlin-dp/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/6012024-1235-maximum-profit-in-job?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/V0pMlKRWRQU)
![image.png](/assets/leetcode_daily_images/17e1b78f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/463

#### Problem TLDR

Max profit in non-intersecting jobs given startTime[], endTime[] and profit[].

#### Intuition

Start with sorting jobs by the `startTime`. Then let's try to find a subproblem: consider the only last element - it has maximum profit in itself. Then, move one index left: now, if we take the element, we must drop all the intersected jobs. Given that logic, there is a Dynamic Programming recurrence: `dp[i] = max(dp[i + 1], profit[i] + dp[next])`.

The tricky part is how to faster find the `next` non-intersecting position: we can use the Binary Search

#### Approach

Try to solve the problem for examples, there are only several ways you could try: greedy or dp. After 1 hour, use the hints.

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun jobScheduling(startTime: IntArray, endTime: IntArray, profit: IntArray): Int {
    val inds = startTime.indices.sortedBy { startTime[it] }
    val dp = IntArray(inds.size + 1)
    for (i in inds.indices.reversed()) {
      var lo = i + 1
      var hi = inds.lastIndex
      while (lo <= hi) {
        val m = lo + (hi - lo) / 2
        if (endTime[inds[i]] > startTime[inds[m]]) lo = m + 1 else hi = m - 1
      }
      dp[i] = max(dp[i + 1], profit[inds[i]] + dp[lo])
    }
    return dp[0]
  }

```

