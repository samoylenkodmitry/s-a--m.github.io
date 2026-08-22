---
layout: leetcode-entry
title: "3622. Check Divisibility by Digit Sum and Product"
permalink: "/leetcode/problem/2026-08-22-3622-check-divisibility-by-digit-sum-and-product/"
leetcode_ui: true
entry_slug: "2026-08-22-3622-check-divisibility-by-digit-sum-and-product"
---

[3622. Check Divisibility by Digit Sum and Product](https://leetcode.com/problems/check-divisibility-by-digit-sum-and-product/solutions/8475896/kotlin-rust-by-samoylenkodmitry-gd8s/) easy
[substack](https://dmitriisamoilenko.substack.com/p/22082026-3622-check-divisibility?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/RQlXd3TXWpA)

https://dmitrysamoylenko.com/leetcode/

![22.08.2026.webp](/assets/leetcode_daily_images/22.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1459

#### Problem TLDR

Is divisible by sum + product of digits

#### Intuition

Brute force.

However, as a joke, this is accepted too
```
    fun checkDivisibility(n: Int)='0' in "$n" && n%
    "$n".sumOf{it-'0'}<1||n in 19..99 step 10||n in setOf(42,111111,794556,979968)
```
The problem has an interesting distribution pattern of the product influence to divisibility.
![Code_Generated_Image (2).png](https://assets.leetcode.com/users/images/3bc3ecf5-3248-41e8-a64d-9c1cbc5efc4a_1787394603.9273818.png)

#### Approach

* just do brute forcec

#### Complexity

- Time complexity:
$$O(logn)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun checkDivisibility(n: Int) = 1>n%"$n"
    .map {it-'0'}.run{sum()+reduce{a,b->a*b}}
```
```rust
    pub fn check_divisibility(n: i32) -> bool {
        let (mut t, mut s, mut p) = (n, 0, 1);
        while t > 0 { s += t % 10; p *= t % 10; t /= 10 }
        n % (s + p) == 0
    }
```

