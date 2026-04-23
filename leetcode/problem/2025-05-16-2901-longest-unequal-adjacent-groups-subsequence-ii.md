---
layout: leetcode-entry
title: "2901. Longest Unequal Adjacent Groups Subsequence II"
permalink: "/leetcode/problem/2025-05-16-2901-longest-unequal-adjacent-groups-subsequence-ii/"
leetcode_ui: true
entry_slug: "2025-05-16-2901-longest-unequal-adjacent-groups-subsequence-ii"
---

[2901. Longest Unequal Adjacent Groups Subsequence II](https://leetcode.com/problems/longest-unequal-adjacent-groups-subsequence-ii/description) medium
[blog post](https://leetcode.com/problems/longest-unequal-adjacent-groups-subsequence-ii/solutions/6749642/kotlin-rust-by-samoylenkodmitry-zwok/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16052025-2901-longest-unequal-adjacent?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/BR9_MW6AycI)
![1.webp](/assets/leetcode_daily_images/6c332dff.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/990

#### Problem TLDR

Longest subsequence humming distance 1, alterating groups #medium #dp

#### Intuition

The naive DP works: consider each position, take or not. Cache by the current and the previous taken position.

#### Approach

Optimizations:

* only the previous position matters for the cache; search for the tail after it
* rewrite DFS into iterative backwards `for`
* then reverse the iterations: for each `i` consider `0..i-1` prefixes, choose the longest
* use the `parents` array to save only the lengths into `dp` array

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 166ms
    fun getWordsInLongestSubsequence(w: Array<String>, g: IntArray): List<String> {
        val dp = Array(w.size + 1) { listOf<String>() }
        for (i in 0..w.size) for (j in 0..<i)
            if (i == w.size || g[i] != g[j] && w[i].length == w[j].length &&
                w[i].indices.count { w[i][it] != w[j][it] } < 2)
                if (dp[j].size + 1 > dp[i].size) dp[i] = dp[j] + w[j]
        return dp[w.size]
    }

```
```kotlin

// 44ms https://leetcode.com/problems/longest-unequal-adjacent-groups-subsequence-ii/submissions/1635364675
    fun getWordsInLongestSubsequence(w: Array<String>, g: IntArray): List<String> {
        val dp = Array(w.size) { listOf<String>() }
        for (i in w.indices) {
            val wi = w[i]; val gi = g[i]
            for (j in 0..<i) if (g[i] != g[j] && wi.length == w[j].length) {
                val wj = w[j]; var c = 0
                for (k in wi.indices) {
                    if (wi[k] != wj[k]) c++
                    if (c > 1) break
                }
                if (c < 2 && dp[j].size + 1 > dp[i].size) dp[i] = dp[j] + wj
            }
        }
        var res = listOf<String>()
        for (j in w.indices) if (dp[j].size + 1 > res.size) res = dp[j] + w[j]
        return res
    }

```
```kotlin

// 31ms https://leetcode.com/problems/longest-unequal-adjacent-groups-subsequence-ii/submissions/1635402651
    fun getWordsInLongestSubsequence(w: Array<String>, g: IntArray): Array<String?> {
        val dp = IntArray(w.size + 1); val p = IntArray(dp.size)
        for (i in w.indices) {
            val wi = w[i]; val gi = g[i]
            for (j in 0..<i) if (g[i] != g[j] && wi.length == w[j].length) {
                val wj = w[j]; var c = 0
                for (k in wi.indices) {
                    if (wi[k] != wj[k]) c++
                    if (c > 1) break
                }
                if (c < 2 && dp[j] + 1 > dp[i]) { dp[i] = dp[j] + 1; p[i] = j + 1 }
            }
        }
        for (j in w.indices) if (dp[j] + 1 > dp[w.size]) { dp[w.size] = dp[j] + 1; p[w.size] = j + 1 }
        val res = Array<String?>(dp[w.size]) { null }; var x = w.size; var k = res.size - 1
        while (p[x] > 0) { res[k--] = w[p[x] - 1]; x = p[x] - 1 }
        return res
    }

```
```rust

// 304ms
    pub fn get_words_in_longest_subsequence(w: Vec<String>, g: Vec<i32>) -> Vec<String> {
        let mut dp = vec![Vec::<String>::new(); w.len() + 1];
        for i in 0..dp.len() { for j in 0..i { if
            i == w.len() || g[i] != g[j] && w[i].len() == w[j].len() &&
            w[i].bytes().zip(w[j].bytes()).filter(|(a, b)| a != b).count() < 2 {
            if dp[j].len() + 1 > dp[i].len() { let mut s = dp[j].clone(); s.push(w[j].clone()); dp[i] = s }}}}
        dp[w.len()].clone()
    }

```
```rust

// 7ms https://leetcode.com/problems/longest-unequal-adjacent-groups-subsequence-ii/submissions/1635396131
    pub fn get_words_in_longest_subsequence(w: Vec<String>, g: Vec<i32>) -> Vec<String> {
        let mut dp = vec![0; w.len() + 1]; let mut p = vec![None; dp.len()];
        for i in 1..=w.len() { for j in 0..i {
            if (i == w.len() || (g[i] != g[j] && w[i].len() == w[j].len()
                && w[i].bytes().zip(w[j].bytes()).filter(|(a, b)| a != b).count() < 2))
            && dp[j] + 1 > dp[i] { dp[i] = dp[j] + 1; p[i] = Some(j); }
        }}
        let (mut res, mut i) = (Vec::with_capacity(dp[w.len()]), w.len());
        while let Some(j) = p[i] { res.push(w[j].clone()); i = j; }
        res.reverse(); res
    }

```
```c++

// 59ms
    vector<string> getWordsInLongestSubsequence(vector<string>& w, vector<int>& g) {
        int n = w.size(); vector<int> dp(n+1), p(n+1, -1); vector<string> r;
        for (int i = 1; i <= n; ++i) for (int j = 0; j < i; ++j) {
                bool ok = i == n
                    || (g[i] != g[j] && size(w[i]) == size(w[j])
                        && [&]{ int c=0;
                            for (int k=0; k<size(w[i]); ++k)
                                if (w[i][k]!=w[j][k] && ++c>=2) return false;
                            return true;
                        }());
                if (ok && dp[j] + 1 > dp[i]) dp[i] = dp[j] + 1, p[i] = j;
            }
        for (int i = n; p[i] != -1; i = p[i]) r.push_back(w[p[i]]);
        reverse(r.begin(), r.end()); return r;
    }

```

