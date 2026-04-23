---
layout: leetcode-entry
title: "1437. Check If All 1's Are at Least Length K Places Away"
permalink: "/leetcode/problem/2025-11-17-1437-check-if-all-1-s-are-at-least-length-k-places-away/"
leetcode_ui: true
entry_slug: "2025-11-17-1437-check-if-all-1-s-are-at-least-length-k-places-away"
---

[1437. Check If All 1's Are at Least Length K Places Away](https://leetcode.com/problems/check-if-all-1s-are-at-least-length-k-places-away/description) easy
[blog post](https://leetcode.com/problems/check-if-all-1s-are-at-least-length-k-places-away/solutions/7354571/kotlin-rust-by-samoylenkodmitry-fjcr/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17112025-1437-check-if-all-1s-are?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/8DrG_10KmYQ)

![dc32b130-129d-412c-b438-b0e73b728df3 (1).webp](/assets/leetcode_daily_images/3978a9bf.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1176

#### Problem TLDR

All ones k-distant #easy

#### Intuition

Count zeros in-between.

#### Approach

* we can write it with 1 extra variable

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 3ms
    fun kLengthApart(nums: IntArray, k: Int): Boolean {
        var z = k
        return nums.all { n -> (n < 1 || z >= k).also { z = (1-n)*(z+1-n)}}
    }
```
```rust
// 0ms
    pub fn k_length_apart(n: Vec<i32>, k: i32) -> bool {
        let mut z = k;
        n.iter().all(|&n|{ let r = n < 1 || z >= k; z = (1-n)*(z+1-n); r})
    }
```

