---
layout: leetcode-entry
title: "3350. Adjacent Increasing Subarrays Detection II"
permalink: "/leetcode/problem/2025-10-15-3350-adjacent-increasing-subarrays-detection-ii/"
leetcode_ui: true
entry_slug: "2025-10-15-3350-adjacent-increasing-subarrays-detection-ii"
---

[3350. Adjacent Increasing Subarrays Detection II](https://leetcode.com/problems/adjacent-increasing-subarrays-detection-ii/description/) medium
[blog post](https://leetcode.com/problems/adjacent-increasing-subarrays-detection-ii/solutions/7276737/kotlin-rust-by-samoylenkodmitry-hew7/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15102025-3350-adjacent-increasing?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/DhwwuGBKqyg)

![8344491e-361e-498a-a277-dc8168c5c52a (1).webp](/assets/leetcode_daily_images/550bbb24.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1143

#### Problem TLDR

Max k of adjucent increasing k-windows #medium #counting

#### Intuition

In a single iteration, count of increasing numbers. Drop on non-increasing. Compare with previous.

#### Approach

* can use `chunk_by` and `windows` in Rust

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$, or O(n)

#### Code

```kotlin

// 870ms
    fun maxIncreasingSubarrays(n: List<Int>): Int {
        var a = 0; var b = 0; var p = 0
        return n.maxOf { n ->
            if (n > p) ++a else { b = a; a = 1 }
            p = n; max(a/2, min(a, b))
        }
    }

```
```rust

// 26ms
    pub fn max_increasing_subarrays(n: Vec<i32>) -> i32 {
        once(0).chain(n.chunk_by(|a,b| a < b).map(|c| c.len() as i32)).collect::<Vec<_>>()
        .windows(2).map(|c| c[0].min(c[1]).max(c[1]/2)).max().unwrap()
    }

```

