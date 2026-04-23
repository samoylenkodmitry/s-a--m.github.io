---
layout: leetcode-entry
title: "2140. Solving Questions With Brainpower"
permalink: "/leetcode/problem/2023-05-12-2140-solving-questions-with-brainpower/"
leetcode_ui: true
entry_slug: "2023-05-12-2140-solving-questions-with-brainpower"
---

[2140. Solving Questions With Brainpower](https://leetcode.com/problems/solving-questions-with-brainpower/description/) medium

```kotlin

fun mostPoints(questions: Array<IntArray>): Long {
    val dp = LongArray(questions.size)
    for (i in questions.lastIndex downTo 0) {
        val (points, skip) = questions[i]
        val tail = if (i + skip + 1 > questions.lastIndex) 0 else dp[i + skip + 1]
        val notTake = if (i + 1 > questions.lastIndex) 0 else dp[i + 1]
        dp[i] = maxOf(points + tail, notTake)
    }
    return dp[0]
}

```

or minified golf version

```

fun mostPoints(questions: Array<IntArray>): Long {
    val dp = HashMap<Int, Long>()
    for ((i, q) in questions.withIndex().reversed())
    dp[i] = maxOf(q[0] + (dp[i + q[1] + 1]?:0), dp[i + 1]?:0)
    return dp[0]?:0
}

```

[blog post](https://leetcode.com/problems/solving-questions-with-brainpower/solutions/3514521/kotlin-dp/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-12052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/210
#### Intuition
If we go from the tail, for each element we are interested only on what happens to the `right` from it. Prefix of the array is irrelevant, when we're starting from the element `i`, because we sure know, that we are taking it and not skipping.
Given that, dynamic programming equation is:
$$dp_i = max(points_i + dp_{i+1+skip_i}, dp_{i+1})$$, where `dp` is the `mostPoints` starting from position `i`.

#### Approach
Let's implement a bottom-up solution.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

