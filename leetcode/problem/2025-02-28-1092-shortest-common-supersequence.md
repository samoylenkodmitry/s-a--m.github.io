---
layout: leetcode-entry
title: "1092. Shortest Common Supersequence"
permalink: "/leetcode/problem/2025-02-28-1092-shortest-common-supersequence/"
leetcode_ui: true
entry_slug: "2025-02-28-1092-shortest-common-supersequence"
---

[1092. Shortest Common Supersequence](https://leetcode.com/problems/shortest-common-supersequence/description/) hard
[blog post](https://leetcode.com/problems/shortest-common-supersequence/solutions/6476850/kotlin-rust-by-samoylenkodmitry-grvc/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28022025-1092-shortest-common-supersequence?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/rhKF0HtCQgQ)
![1.webp](/assets/leetcode_daily_images/da10c2b0.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/909

#### Problem TLDR

Shortest string with both subsequences #hard #lcs #dp

#### Intuition

Naive dp has O(n^3) time complexity and gives TLE.
I used the hint: use the longest common subsequence.

Let's observe how to use it:

```j

    // 221100     21100
    // abacad     becec
    //  * *       * *
    //
    //   i01  2 3  4 5
    //    22  1 1  0 0
    //    ab  a.c  a.d
    //     *    *
    //     b  .ec  .ec
    //     2   11   00
    //    j0   12   34
    //
    // abac       cab
    // **          **
    //   abac
    //   **
    //  cab

```
At each decision-making point we compare the next lcs of two variants and peek the longest. Longest common = shortest result.

#### Approach

* use recursion + cache or bottom-up dp
* pay attention to the DP, it should solve the task of maximizing the common chars (I failed to pay attention to this and wasted 20 minutes)

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

    fun shortestCommonSupersequence(str1: String, str2: String) = buildString {
        val dp = HashMap<Pair<Int, Int>, Int>(); var i = 0; var j = 0
        fun dfs(i: Int, j: Int): Int = dp.getOrPut(i to j) { when {
            i == str1.length || j == str2.length -> 0
            str1[i] == str2[j] -> 1 + dfs(i + 1, j + 1)
            else -> max(dfs(i + 1, j), dfs(i, j + 1)) }}
        while (i < str1.length || j < str2.length)
            if (i == str1.length) append(str2[j++]) else if (j == str2.length) append(str1[i++])
            else if (str1[i] == str2[j]) { append(str1[i++]); j++ }
            else if (dfs(i + 1, j) > dfs(i, j + 1)) append(str1[i++]) else append(str2[j++])
    }

```
```rust

    pub fn shortest_common_supersequence(a: String, b: String) -> String {
        let (a, b, mut i, mut j, mut res) = (a.as_bytes(), b.as_bytes(), 0, 0, vec![]);
        let mut dp = vec![vec![0; b.len() + 1]; a.len() + 1];
        for i in (0..a.len()).rev() { for j in (0..b.len()).rev() { dp[i][j] =
            if a[i] == b[j] { 1 + dp[i + 1][j + 1] } else { dp[i][j + 1].max(dp[i + 1][j]) }}}
        while i < a.len() || j < b.len() {
            if i == a.len() { res.push(b[j]); j += 1 } else if j == b.len() { res.push(a[i]); i += 1}
            else if a[i] == b[j] { res.push(a[i]); i += 1; j += 1 }
            else if dp[i + 1][j] > dp[i][j + 1] { res.push(a[i]); i += 1 }
            else { res.push(b[j]); j += 1 }}
        String::from_utf8(res).unwrap()
    }

```
```c++

string shortestCommonSupersequence(string a, string b) {
    int m = size(a), n = size(b), i = 0, j = 0; string res;
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));
    for (int i = m - 1; i >= 0; i--) for (int j = n - 1; j >= 0; j--)
        dp[i][j] = a[i] == b[j] ? 1 + dp[i + 1][j + 1] : max(dp[i + 1][j], dp[i][j + 1]);
    while (i < m || j < n)
        if (i == m) res += b[j++]; else if (j == n) res += a[i++];
        else if (a[i] == b[j]) res += a[i++], j++;
        else if (dp[i + 1][j] > dp[i][j + 1]) res += a[i++]; else res += b[j++];
    return res;
}

```

