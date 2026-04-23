---
layout: leetcode-entry
title: "3005. Count Elements With Maximum Frequency"
permalink: "/leetcode/problem/2025-09-22-3005-count-elements-with-maximum-frequency/"
leetcode_ui: true
entry_slug: "2025-09-22-3005-count-elements-with-maximum-frequency"
---

[3005. Count Elements With Maximum Frequency](https://leetcode.com/problems/count-elements-with-maximum-frequency/description/) easy
[blog post](https://leetcode.com/problems/count-elements-with-maximum-frequency/solutions/7213082/kotlin-rust-by-samoylenkodmitry-scof/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22092025-3005-count-elements-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/bnI54WtsBgs)

![1.webp](/assets/leetcode_daily_images/ff28f409.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1120

#### Problem TLDR

Count max-freq elements #easy #counting

#### Intuition

Maintain frequency map. Count on-line or in the second iteration.

#### Approach

* for n=100 brute force is accepted

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 13ms
    fun maxFrequencyElements(n: IntArray) =
    n.groupBy{it}.values.map{it.size}.run {max()*count{it==max()}}

```
```kotlin

// 1ms
    fun maxFrequencyElements(n: IntArray): Int {
        var res = 0; var maxF = 0; val f = IntArray(101)
        for (x in n) if (++f[x] > maxF) { maxF = f[x]; res = 1 }
                     else if (f[x] == maxF) ++res;
        return res * maxF
    }

```
```rust

// 0ms
    pub fn max_frequency_elements(mut n: Vec<i32>) -> i32 {
        n.sort_unstable(); n.chunk_by(|a, b| a == b)
        .fold((0, 0, 0), |r, c| if c.len() > r.0 { (c.len(), 1, c.len())}
            else if c.len() == r.0 { (r.0, r.1 + 1, r.0 * (r.1+1))} else { r }).2 as _
    }

```

