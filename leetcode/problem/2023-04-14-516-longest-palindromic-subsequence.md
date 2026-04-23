---
layout: leetcode-entry
title: "516. Longest Palindromic Subsequence"
permalink: "/leetcode/problem/2023-04-14-516-longest-palindromic-subsequence/"
leetcode_ui: true
entry_slug: "2023-04-14-516-longest-palindromic-subsequence"
---

[516. Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/description/) medium

```kotlin

fun longestPalindromeSubseq(s: String): Int {
    // b + abcaba
    // b + ab_ab_
    // b + a_cab_
    // acbbc + a -> [acbbc]a x[from]==x[to]?1 + p[from+1][to-1]
    val p = Array(s.length) { Array(s.length) { 0 } }
    for (i in s.lastIndex downTo 0) p[i][i] = 1
    for (from in s.lastIndex - 1 downTo 0)
    for (to in from + 1..s.lastIndex)
    p[from][to] = if (s[from] == s[to]) {
        2 + if (to == from + 1) 0 else p[from + 1][to - 1]
    } else {
        maxOf(p[from][to - 1], p[from + 1][to])
    }
    return p[0][s.lastIndex]
}

```

[blog post](https://leetcode.com/problems/longest-palindromic-subsequence/solutions/3415189/kotlin-dp/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-14042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/180
#### Intuition
Simple DFS would not work as it will take $$O(2^n)$$ steps.
Consider the sequence: `acbbc` and a new element `a`. The already existing largest palindrome is `cbbc`. When adding a new element, we do not care about what is inside between `a..a`, just the largest value of it.
So, there is a DP equation derived from this observation: $$p[i][j] = eq ? 2 + p[i+1][j-1] : max(p[i][j-1], p[i+1][j])$$.
#### Approach
For cleaner code:
* precompute `p[i][i] = 1`
* exclude `0` and `lastIndex` from iteration
* start with `to = from + 1`
#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n^2)$$

