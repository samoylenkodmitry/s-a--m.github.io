---
layout: leetcode-entry
title: "3487. Maximum Unique Subarray Sum After Deletion"
permalink: "/leetcode/problem/2025-07-25-3487-maximum-unique-subarray-sum-after-deletion/"
leetcode_ui: true
entry_slug: "2025-07-25-3487-maximum-unique-subarray-sum-after-deletion"
---

[3487. Maximum Unique Subarray Sum After Deletion](https://leetcode.com/problems/maximum-unique-subarray-sum-after-deletion/description/) easy
[blog post](https://leetcode.com/problems/maximum-unique-subarray-sum-after-deletion/solutions/7002148/kotlin-rust-by-samoylenkodmitry-sbd0/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25072025-3487-maximum-unique-subarray?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/D2BL5NkhD40)
![1.webp](/assets/leetcode_daily_images/c0b7203c.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1060

#### Problem TLDR

Max non-empty uniq subsequence sum #easy

#### Intuition

Remove all negatives, dedup all positives, then sum.

#### Approach

* careful with `non-empty`, should take 1 negative

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 23ms
    fun maxSum(n: IntArray) =
        n.max().takeIf { it < 0 } ?:
        n.filter { it > 0 }.distinct().sum()

```
```kotlin

// 12ms
    fun maxSum(n: IntArray): Int {
        val f = IntArray(101)
        for (x in n) if (x >= 0) f[x] = x
        return n.max().takeIf { it < 0 } ?: f.sum()
    }

```
```rust

// 0ms
    pub fn max_sum(mut n: Vec<i32>) -> i32 {
        n.sort_unstable(); n.dedup();
        if n[n.len() - 1] < 0 { n[n.len() - 1] }
        else { n.retain(|&x| x > 0); n.into_iter().sum() }
    }

```
```c++

// 1ms
    int maxSum(vector<int>& n) {
        int f[101]={}, m = n[0], s = 0;
        for (int x: n) m = max(m, x), s -= x < 0 ? 0 : (f[x] - (f[x] = x));
        return m < 0 ? m : s;
    }

```

