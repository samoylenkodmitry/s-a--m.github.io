---
layout: leetcode-entry
title: "1128. Number of Equivalent Domino Pairs"
permalink: "/leetcode/problem/2025-05-04-1128-number-of-equivalent-domino-pairs/"
leetcode_ui: true
entry_slug: "2025-05-04-1128-number-of-equivalent-domino-pairs"
---

[1128. Number of Equivalent Domino Pairs](https://leetcode.com/problems/number-of-equivalent-domino-pairs/description/) easy
[blog post](https://leetcode.com/problems/number-of-equivalent-domino-pairs/solutions/6713083/kotlin-rust-by-samoylenkodmitry-v67f/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04052025-1128-number-of-equivalent?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/2z3xSB9UmrM)
![1.webp](/assets/leetcode_daily_images/e85c55cf.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/978

#### Problem TLDR

Dominoes pairs #easy #hash

#### Intuition

The brute-force O(n^2) is accepted.
More optimal O(n) is the counting pattern: count visited pairs, each new will pair with previous count.

#### Approach

* we can try to CPU-branching optimize by using a symmetric cache key `a * b + 10 * (a + b)`
* otherwise, space-optimizations are possible too: we have a symmetric matrix of 9x9 and a total of 45 uniq keys

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 4ms
    fun numEquivDominoPairs(ds: Array<IntArray>): Int {
        val m = Array(10) { IntArray(10) }
        return ds.sumOf { (a, b) -> m[min(a, b)][max(a, b)]++ }
    }

```
```kotlin

// 2ms https://leetcode.com/problems/number-of-equivalent-domino-pairs/submissions/1625039917
fun numEquivDominoPairs(ds: Array<IntArray>): Int {
    val m = IntArray(262); var c = 0
    for ((a, b) in ds) c += m[a * b + 10 * (a + b)]++
    return c
}

```
```rust

// 0ms
    pub fn num_equiv_domino_pairs(ds: Vec<Vec<i32>>) -> i32 {
        let mut m = [0; 100];
        ds.iter().map(|d| {
            let k = (d[0].min(d[1]) * 10 + d[0].max(d[1])) as usize;
            let c = m[k]; m[k] += 1; c
        }).sum()
    }

```
```c++

// 0ms
    int numEquivDominoPairs(vector<vector<int>>& ds) {
        int m[100], c = 0;
        for (auto& d: ds) c += m[min(d[0], d[1]) * 10 + max(d[0], d[1])]++;
        return c;
    }

```

