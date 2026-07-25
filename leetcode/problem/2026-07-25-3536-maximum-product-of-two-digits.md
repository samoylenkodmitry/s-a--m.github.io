---
layout: leetcode-entry
title: "3536. Maximum Product of Two Digits"
permalink: "/leetcode/problem/2026-07-25-3536-maximum-product-of-two-digits/"
leetcode_ui: true
entry_slug: "2026-07-25-3536-maximum-product-of-two-digits"
---

[3536. Maximum Product of Two Digits](https://leetcode.com/problems/maximum-product-of-two-digits/solutions/8418932/kotlin-rust-by-samoylenkodmitry-ylz1/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25072026-3536-maximum-product-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/xvdJKKuH04w)

https://dmitrysamoylenko.com/leetcode/

![25.07.2026.webp](/assets/leetcode_daily_images/25.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1431

#### Problem TLDR

Product of two largest digits in n

#### Intuition

Convert to string. Sort.

#### Approach

* Rust itertools has k_largest

#### Complexity

- Time complexity:
$$O(logn*log log n)$$

- Space complexity:
$$O(logn)$$

#### Code

```kotlin
    fun maxProduct(n: Int) =
    "$n".map{it-'0'}.sorted().takeLast(2).let{it[0]*it[1]}
```
```rust
    pub fn max_product(n: i32) -> i32 {
        format!("0{n}").bytes().k_largest(2).map(|b| b as i32 - 48).product()
    }
```

