---
layout: leetcode-entry
title: "3349. Adjacent Increasing Subarrays Detection I"
permalink: "/leetcode/problem/2025-10-14-3349-adjacent-increasing-subarrays-detection-i/"
leetcode_ui: true
entry_slug: "2025-10-14-3349-adjacent-increasing-subarrays-detection-i"
---

[3349. Adjacent Increasing Subarrays Detection I](https://leetcode.com/problems/adjacent-increasing-subarrays-detection-i/description) medium
[blog post](https://leetcode.com/problems/adjacent-increasing-subarrays-detection-i/solutions/7274188/kotlin-rust-by-samoylenkodmitry-vdz6/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14102025-3349-adjacent-increasing?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/gNzcOKsRCPQ)

![6a711a37-86e3-4929-b20f-737b09eb2b2f (1) (1).webp](/assets/leetcode_daily_images/0ae9a7b0.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1142

#### Problem TLDR

Any increasing consequent windows size k #easy

#### Intuition

Just brute-force every index.
O(1) memory solution: count increasings, keep previous and current, check

#### Approach

* corner case: single 2k increasing chunk

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, can be O(1)

#### Code

```kotlin

// 379ms
    fun hasIncreasingSubarrays(n: List<Int>, k: Int) = n.indices
        .any { i -> fun List<Int>.g()=this==sorted()&&toSet().size==k;
            n.slice(i..<min(n.size,i+k)).g() &&
            n.slice(min(n.size-1,i+k)..<min(n.size,i+k+k)).g()
        }

```
```rust

// 5ms
    pub fn has_increasing_subarrays(n: Vec<i32>, k: i32) -> bool {
        once(0).chain(n.chunk_by(|a,b| a < b).map(|c|c.len() as i32))
        .collect::<Vec<_>>().windows(2).any(|c| c[0].min(c[1]).max(c[1]/2) >= k)
    }

```

