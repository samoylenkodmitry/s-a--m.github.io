---
layout: leetcode-entry
title: "3754. Concatenate Non-Zero Digits and Multiply by Sum I"
permalink: "/leetcode/problem/2026-07-07-3754-concatenate-non-zero-digits-and-multiply-by-sum-i/"
leetcode_ui: true
entry_slug: "2026-07-07-3754-concatenate-non-zero-digits-and-multiply-by-sum-i"
---

[3754. Concatenate Non-Zero Digits and Multiply by Sum I](https://leetcode.com/problems/concatenate-non-zero-digits-and-multiply-by-sum-i/solutions/8381418/kotlin-rust-by-samoylenkodmitry-cp5z/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07072026-3754-concatenate-non-zero?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Sxm72icB-yI)

https://dmitrysamoylenko.com/leetcode/

![07.07.2026.webp](/assets/leetcode_daily_images/07.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1413

#### Problem TLDR

Sum digits * concat digits

#### Intuition

Sum them and concat them with strings. Or go from tail with match.

#### Approach

* can be done in two phases for shorter code

#### Complexity

- Time complexity:
$$O(logn)$$

- Space complexity:
$$O(logn)$$

#### Code

```kotlin
    fun sumAndMultiply(n: Int) =
    ("0"+"$n".filter{it>'0'}).toLong()*"$n".sumOf{it-'0'}
```
```rust
    pub fn sum_and_multiply(mut n: i32) -> i64 {
        let(mut v,mut s,mut m)=(0,0,1);
        while n>0{let d=(n%10)as i64;s+=d;if d>0{v+=d*m;m*=10}n/=10}v*s
    }
```

