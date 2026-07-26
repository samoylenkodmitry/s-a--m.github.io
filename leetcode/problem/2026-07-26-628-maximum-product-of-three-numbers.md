---
layout: leetcode-entry
title: "628. Maximum Product of Three Numbers"
permalink: "/leetcode/problem/2026-07-26-628-maximum-product-of-three-numbers/"
leetcode_ui: true
entry_slug: "2026-07-26-628-maximum-product-of-three-numbers"
---

[628. Maximum Product of Three Numbers](https://leetcode.com/problems/maximum-product-of-three-numbers/solutions/8421251/kotlin-rust-by-samoylenkodmitry-ym93/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26072026-628-maximum-product-of-three?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/DDMgAKSD1V0)

https://dmitrysamoylenko.com/leetcode/

![26.07.2026.webp](/assets/leetcode_daily_images/26.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1432

#### Problem TLDR

Max product of 3 numbers

#### Intuition

Take 3 largest and compare with taking two smallest and a one largest.

#### Approach

* Rust itertools has k_largest, s_smallest (the results are sorted differently)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun maximumProduct(n: IntArray) = run { n.sort()
    val l = n.size-1; max(n[0]*n[1]*n[l], n[l]*n[l-1]*n[l-2])}
```
```rust
    pub fn maximum_product(n: Vec<i32>) -> i32 {
        let (a,b) = n.iter().copied().k_smallest(2).collect_tuple().unwrap();
        let (z,y,x) = n.iter().copied().k_largest(3).collect_tuple().unwrap();
        (a*b*z).max(x*y*z)
    }
```

