---
layout: leetcode-entry
title: "2140. Solving Questions With Brainpower"
permalink: "/leetcode/problem/2025-04-01-2140-solving-questions-with-brainpower/"
leetcode_ui: true
entry_slug: "2025-04-01-2140-solving-questions-with-brainpower"
---

[2140. Solving Questions With Brainpower](https://leetcode.com/problems/solving-questions-with-brainpower/description) medium
[blog post](https://leetcode.com/problems/solving-questions-with-brainpower/solutions/6602476/kotlin-rust-by-samoylenkodmitry-dg2h/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01042025-2140-solving-questions-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/zW2tQpPD-lA)

![1.webp](/assets/leetcode_daily_images/4a7e0f5a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/945

#### Problem TLDR

Max sum of (points, skip_next) pairs #medium #dp

#### Intuition

At each position make a decision: take or skip. We should know before-hand the oprimal result of (current + skip) position. Go from the tail, or do a DFS + cache.

#### Approach

* use `size + 1` for dp array to simplify the logic

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun mostPoints(q: Array<IntArray>): Long {
        val dp = LongArray(q.size + 1)
        for (i in q.size - 1 downTo 0) dp[i] = max(
            dp[min(q.size, i + q[i][1] + 1)] + q[i][0], dp[i + 1])
        return dp[0]
    }

```
```kotlin

    val dp = HashMap<Int, Long>()
    fun mostPoints(q: Array<IntArray>, i: Int = 0): Long = if (i >= q.size) 0 else
    dp.getOrPut(i) { max(mostPoints(q, i + q[i][1] + 1) + q[i][0], mostPoints(q, i + 1)) }

```
```rust

    pub fn most_points(q: Vec<Vec<i32>>) -> i64 {
        let mut dp = vec![0; q.len() + 1];
        for i in (0..q.len()).rev() { dp[i] = dp[i + 1].max(
            dp[q.len().min(i + 1 + q[i][1] as usize)] + q[i][0] as i64 )}
        dp[0]
    }

```
```c++

    long long mostPoints(vector<vector<int>>& q) {
        int n = size(q); vector<long long> dp(n + 1, 0);
        for (int i = n - 1; i >= 0; --i) dp[i] = max(dp[i + 1],
            dp[min(n, i + 1 + q[i][1])] + q[i][0]);
        return dp[0];
    }

```

