---
layout: leetcode-entry
title: "2900. Longest Unequal Adjacent Groups Subsequence I"
permalink: "/leetcode/problem/2025-05-15-2900-longest-unequal-adjacent-groups-subsequence-i/"
leetcode_ui: true
entry_slug: "2025-05-15-2900-longest-unequal-adjacent-groups-subsequence-i"
---

[2900. Longest Unequal Adjacent Groups Subsequence I](https://leetcode.com/problems/longest-unequal-adjacent-groups-subsequence-i/description/) easy
[blog post](https://leetcode.com/problems/longest-unequal-adjacent-groups-subsequence-i/solutions/6746113/kotlin-rust-by-samoylenkodmitry-zse0/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15052025-2900-longest-unequal-adjacent?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Ufns428OJeA)
![1.webp](/assets/leetcode_daily_images/5f01df27.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/989

#### Problem TLDR

Longest subsequence alterating g #easy #gready

#### Intuition

Taking the first is always the optimal greedy strategy:

```j
    // 101
    // 010
```

#### Approach

* we actually don't have to remember the last `g[i]`, compare with the previous
* O(n^2), O(1) memory solution: remove from `words` (c++)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 22ms
    fun getLongestSubsequence(w: Array<String>, g: IntArray) =
        g.indices.filter { it < 1 || g[it - 1] != g[it] }.map(w::get)

```
```kotlin

// 15ms
    fun getLongestSubsequence(w: Array<String>, g: IntArray) = buildList {
        for (i in g.indices) if (i < 1 || g[i] != g[i - 1]) this += w[i]
    }

```
```kotlin

// 1ms
    fun getLongestSubsequence(w: Array<String>, g: IntArray): List<String> {
        val res = ArrayList<String>()
        for (i in g.indices) if (i < 1 || g[i] != g[i - 1]) res += w[i]
        return res
    }

```
```rust

// 0ms
    pub fn get_longest_subsequence(mut w: Vec<String>, g: Vec<i32>) -> Vec<String> {
        for i in (1..g.len()).rev() { if g[i] == g[i - 1] { w.remove(i); }} w
    }

```
```rust

// 0ms
    pub fn get_longest_subsequence(w: Vec<String>, g: Vec<i32>) -> Vec<String> {
       w.into_iter().enumerate().filter(|&(i, _)| i < 1 || g[i - 1] != g[i]).map(|(_, w)| w).collect()
    }

```
```c++

// 0ms
    vector<string> getLongestSubsequence(vector<string>& w, vector<int>& g) {
        for(int i = w.size(); i-- > 1; ) if (g[i] == g[i-1]) w.erase(begin(w) + i);
        return w;
    }

```

