---
layout: leetcode-entry
title: "3606. Coupon Code Validator"
permalink: "/leetcode/problem/2025-12-13-3606-coupon-code-validator/"
leetcode_ui: true
entry_slug: "2025-12-13-3606-coupon-code-validator"
---

[3606. Coupon Code Validator](https://leetcode.com/problems/coupon-code-validator/description/) easy
[blog post](https://leetcode.com/problems/coupon-code-validator/solutions/7410815/kotlin-rust-by-samoylenkodmitry-tkji/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13122025-3606-coupon-code-validator?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/-HSIbCyUbAU)

![fd4f17a4-698f-410e-9dd9-1d3d56a1caf3 (1).webp](/assets/leetcode_daily_images/445165cb.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1203

#### Problem TLDR

Filter a,b,c accroding to rules #easy

#### Intuition

Just read the rules.

#### Approach

* some rules can be hacked around
* the regex in koglin faster than in rust

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 57ms
    fun validateCoupons(c: Array<String>, b: Array<String>, a: BooleanArray) =
    c.indices.filter { a[it] && Regex("\\w+") matches c[it] && b[it][0] in "egpr" }
    .sortedBy { b[it][0] + c[it] }.map { c[it] }
```
```rust
// 0ms
    pub fn validate_coupons(c: Vec<String>, b: Vec<String>, a: Vec<bool>) -> Vec<String> {
        b.iter().map(|b|b.as_bytes()[0]).zip(c).zip(a).filter(|((b,c),a)|
            *a && b"egrp".contains(b) && c != "" && c.chars().all(|c|c.is_alphanumeric() || c == '_'))
        .sorted().map(|((_,c),_)|c).collect()
    }
```

