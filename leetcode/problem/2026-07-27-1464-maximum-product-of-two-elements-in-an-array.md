---
layout: leetcode-entry
title: "1464. Maximum Product of Two Elements in an Array"
permalink: "/leetcode/problem/2026-07-27-1464-maximum-product-of-two-elements-in-an-array/"
leetcode_ui: true
entry_slug: "2026-07-27-1464-maximum-product-of-two-elements-in-an-array"
---

[1464. Maximum Product of Two Elements in an Array](https://leetcode.com/problems/maximum-product-of-two-elements-in-an-array/solutions/8423239/kotlin-rust-by-samoylenkodmitry-il0u/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27072026-1464-maximum-product-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/85Lxmbl2hKY)

https://dmitrysamoylenko.com/leetcode/

![27.07.2026.webp](/assets/leetcode_daily_images/27.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1433

#### Problem TLDR

Max product in array

#### Intuition

Brute-force: iterate in a nested 'for' loops. Accepted.

#### Approach

* we can sort and take two max values
* we can scan and find two max values

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun maxProduct(n: IntArray) =
    n.sorted().run{(last()-1)*(get(n.size-2)-1)}
```
```rust
    pub fn max_product(n: Vec<i32>) -> i32 {
        n.iter().map(|&x|x-1).k_largest(2).product::<i32>()
    }
```

