---
layout: leetcode-entry
title: "2161. Partition Array According to Given Pivot"
permalink: "/leetcode/problem/2025-03-03-2161-partition-array-according-to-given-pivot/"
leetcode_ui: true
entry_slug: "2025-03-03-2161-partition-array-according-to-given-pivot"
---

[2161. Partition Array According to Given Pivot](https://leetcode.com/problems/partition-array-according-to-given-pivot/description/) medium
[blog post](https://leetcode.com/problems/partition-array-according-to-given-pivot/solutions/6489093/kotlin-rust-by-samoylenkodmitry-356j/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03032025-2161-partition-array-according?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/3n4yOm1hwHI)
![1.webp](/assets/leetcode_daily_images/2a2e3a6c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/913

#### Problem TLDR

Partition around p #medium

#### Intuition

In-place solution is possible, but O(nlog(n)).
Otherwise, there are two-pass solution with two pointers tracking, or 3-pass with a single pointer.

#### Approach

* golf it in Kotlin
* in-place in Rust
* 2-pass in C++

#### Complexity

- Time complexity:
$$O(n)$$, or NlogN for sorting

- Space complexity:
$$O(n)$$, or O(1) for in-place sorting

#### Code

```kotlin

    fun pivotArray(n: IntArray, p: Int) =
        n.filter { it < p } + n.filter { it == p } + n.filter { it > p }

```
```rust

    pub fn pivot_array(mut n: Vec<i32>, p: i32) -> Vec<i32> {
        n.sort_by_key(|&x| x.cmp(&p)); n
    }

```
```c++

    vector<int> pivotArray(vector<int>& a, int p) {
        int n = size(a), i = 0; vector<int> r(n);
        for (auto& x: a) if (x < p) r[i++] = x; else n -= x > p;
        while (i < n) r[i++] = p;
        for (auto& x: a) if (x > p) r[i++] = x;
        return r;
    }

```

